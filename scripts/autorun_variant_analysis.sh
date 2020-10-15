#!/bin/bash -i

# Constants
HS38_DIR="/proj/eva-public-PG0/Genome_Data"
DATA_DIR="/mydata"
SPARK_MASTER="spark://vm0:7077"
CANNOLI_SUBMIT=${CANNOLI_HOME}"/exec/cannoli-submit"
ADAM_SUBMIT=${ADAM_HOME}"/exec/adam-submit"
HDFS_PREFIX="hdfs://vm0:9000"
EXECUTOR_MEMORY=50g
DRIVER_MEMORY=50g
REFERENCE="file://"${DATA_DIR}"/"${1}".fa"
DICT="file://"${DATA_DIR}"/"${1}".dict"
FREE_BAYES=${DATA_DIR}"/freebayes/bin/freebayes"
BWA=${DATA_DIR}"/bwa/bwa"
EXTENSIONS=(\
  ".fa" \
  ".fa.amb" \
  ".fa.ann" \
  ".fa.bwt" \
  ".fa.fai" \
  ".fa.pac" \
  ".fa.sa" \
  ".dict")

let NUM_EXECUTORS=${4}
let NUM_CORES=$(nproc)-4

# Input Validation
if [[ $# -lt 4 ]]; then
    echo "Usage: autorun_variant_analysis.sh <hs38|hs38a|hs38DH|hs37|hs37d5> <FASTQ_FILE_1_URL> <FASTQ_FILE_2_URL> <CLUSTER_SIZE> <EXPERIMENT_ID>"
    exit
fi

exp_id=$5
if [[ -z $exp_id ]]; then
        exp_id=$(cat /dev/urandom | tr -dc 'A-Z0-9' | fold -w 6 | head -n 1)
fi
echo "👉 Experiment Id is $exp_id"

echo "👉 Verifying if reference genome is present."
for extension in "${EXTENSIONS[@]}"; do
  if [[ ! -f "${DATA_DIR}/${1}${extension}" ]]; then
    echo "😡 Missing reference genome. Executing setup_reference_genome.sh."
    cd ${DATA_DIR} && ${HOME}/EVA/scripts/setup_reference_genome.sh ${1}
    break
  fi
done

#echo "👉 Downloading the FASTQ files. (Presently only .gz extensions are supported.)"
#wget ${2} -O ${DATA_DIR}/Input_1_${exp_id}.filt.fastq.gz
#wget ${3} -O ${DATA_DIR}/Input_2_${exp_id}.filt.fastq.gz

#echo "👉 Uploading the files to HDFS."
#hdfs dfs -copyFromLocal ${DATA_DIR}/Input_1_${exp_id}.filt.fastq.gz /
#hdfs dfs -copyFromLocal ${DATA_DIR}/Input_2_${exp_id}.filt.fastq.gz /

echo "👉 Downloading FASTQ files and copying to HDFS"
curl -sS ${2} | hdfs dfs -put - /Input_1_${exp_id}.filt.fastq.gz
curl -sS ${3} | hdfs dfs -put - /Input_2_${exp_id}.filt.fastq.gz

echo "👉 Performing variant analysis using Adam."
${HOME}/EVA/scripts/run_variant_analysis_adam.sh ${1} hdfs://vm0:9000/Input_1_${exp_id}.filt.fastq.gz hdfs://vm0:9000/Input_2_${exp_id}.filt.fastq.gz ${4}

echo "👉 Interleaving FASTQ files."
${CANNOLI_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- interleaveFastq hdfs://vm0:9000/Input_1_${exp_id}.filt.fastq.gz hdfs://vm0:9000/Input_2_${exp_id}.filt.fastq.gz ${HDFS_PREFIX}/${exp_id}.ifq

echo "👉 Executing bwa for alignment."
${CANNOLI_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- bwa ${HDFS_PREFIX}/${exp_id}.ifq ${HDFS_PREFIX}/${exp_id}.bam \
    -executable ${BWA} -sample_id $exp_id -index ${REFERENCE} -sequence_dictionary ${DICT} -single -add_files

echo "👉 Sorting and marking duplicates before variant calling."
${ADAM_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- transformAlignments ${HDFS_PREFIX}/${exp_id}.bam ${HDFS_PREFIX}/${exp_id}.bam.adam \
    -mark_duplicate_reads -sort_by_reference_position_and_index

echo "👉 Variant calling using freebayes."
${CANNOLI_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- freebayes ${HDFS_PREFIX}/${exp_id}.bam.adam ${HDFS_PREFIX}/${exp_id}.vcf \
    -executable ${FREE_BAYES} -reference ${REFERENCE} -add_files -single

echo "👉 Copying the vcf to ${DATA_DIR}."
hdfs dfs -copyToLocal ${HDFS_PREFIX}/${exp_id}.vcf ${DATA_DIR}/${exp_id}-fbayes-output.vcf

echo "👉 Compressing the output obtained."
zip -j ${DATA_DIR}/${exp_id}-fbayes-output.vcf.zip ${DATA_DIR}/${exp_id}-fbayes-output.vcf

echo "👉 Deleting HDFS copy of the files."
hdfs dfs -rm -f /Input_?_${exp_id}.filt.fastq.gz
