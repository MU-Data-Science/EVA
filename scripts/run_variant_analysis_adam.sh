#!/usr/bin/env bash

if [[ $# -ne 4 ]]; then
    echo "Usage: run_variant_analysis_adam.sh <hs38|hs38a|hs38DH|hs37|hs37d5> <HDFS_PATH_OF_FASTQ_file1> <HDFS_PATH_OF_FASTQ_file2> <cluster size>"
    exit
fi

SPARK_MASTER="spark://vm0:7077"
CANNOLI_SUBMIT=${CANNOLI_HOME}"/spark2_exec/cannoli-submit"
ADAM_SUBMIT=${ADAM_HOME}"/spark2_exec/adam-submit"
HDFS_PREFIX="hdfs://vm0:9000"
EXECUTOR_MEMORY=50g
DRIVER_MEMORY=50g
INPUT_FILE=mysequence
DATA_DIR="/mydata"
REFERENCE="file://"${DATA_DIR}"/"${1}".fa"
REF_CHECK=${DATA_DIR}"/"${1}."fa"
DICT="file://"${DATA_DIR}"/"${1}".dict"
FREE_BAYES=${DATA_DIR}"/freebayes/bin/freebayes"
BWA=${DATA_DIR}"/bwa/bwa"
GATK=${DATA_DIR}"/gatk-4.1.8.0/gatk"
OUTPUT_PREFIX="VA-"${USER}"-result"
TRANCHE_RESOURCES=(\
  "${DATA_DIR}/hapmap_3.3.hg38.vcf.gz" \
  "${DATA_DIR}/1000G_omni2.5.hg38.vcf.gz" \
  "${DATA_DIR}/1000G_phase1.snps.high_confidence.hg38.vcf.gz")

let NUM_EXECUTORS=${4}
let NUM_CORES=$(nproc)-4

echo "👉 Deleting files..."
hdfs dfs -rm -r ${HDFS_PREFIX}/${INPUT_FILE}*
rm -rvf ${DATA_DIR}/${OUTPUT_PREFIX}-*.vcf*

echo "👉 Validating the cluster."
if [[ ! -f "${REF_CHECK}" ]]; then
    echo "😡 Missing reference genome. Run setup_reference_genome.sh."
    exit
fi

for file in "${TRANCHE_RESOURCES[@]}"; do
  if [[ (! -f "$file") || (! -f "${file}.tbi")]]; then
    echo "😡 Trance Resource ${file} or ${file}.tbi missing."
    exit
  fi
done

date
echo "👉 Interleaving FASTQ files."
${CANNOLI_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- interleaveFastq ${2} ${3} ${HDFS_PREFIX}/${INPUT_FILE}.ifq

echo "👉 Executing bwa for alignment."
${CANNOLI_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- bwaMem ${HDFS_PREFIX}/${INPUT_FILE}.ifq ${HDFS_PREFIX}/${INPUT_FILE}.bam \
    -executable ${BWA} -sample_id mysample -index ${REFERENCE} -sequence_dictionary ${DICT} -single -add_files

echo "👉 Sorting and marking duplicates before variant calling."
${ADAM_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- transformAlignments ${HDFS_PREFIX}/${INPUT_FILE}.bam ${HDFS_PREFIX}/${INPUT_FILE}.bam.adam \
    -mark_duplicate_reads -sort_by_reference_position_and_index

echo "👉 Variant calling using freebayes."
${CANNOLI_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- freebayes ${HDFS_PREFIX}/${INPUT_FILE}.bam.adam ${HDFS_PREFIX}/${INPUT_FILE}.vcf \
    -executable ${FREE_BAYES} -reference ${REFERENCE} -add_files -single

hdfs dfs -copyToLocal ${HDFS_PREFIX}/${INPUT_FILE}.vcf ${HOME}/${OUTPUT_PREFIX}-fbayes-output.vcf
echo "👉 Done with variant analysis. See ${DATA_DIR}/${OUTPUT_PREFIX}-fbayes-output.vcf."

echo "👉 Running Base Quality Score Recalibration."
${HOME}/EVA/scripts/run_BQSR.sh ${1} ${DATA_DIR}/${OUTPUT_PREFIX}-fbayes-output.vcf ${OUTPUT_PREFIX} ${4}

cho "👉 Filtering annotated variants using Convolutional Neural Net."
${GATK} CNNScoreVariants \
  -V ${DATA_DIR}/${OUTPUT_PREFIX}-output-gatk-spark-BQSR-output.vcf \
	-R ${REFERENCE} \
	-O ${DATA_DIR}/${OUTPUT_PREFIX}-cnn-annotated.vcf

echo "👉 Applying tranche filters"
for resource in "${TRANCHE_RESOURCES[@]}"; do
    resources+=( --resource "$resource" )
done

${GATK} FilterVariantTranches \
    -V ${DATA_DIR}/${OUTPUT_PREFIX}-cnn-annotated.vcf \
    -O ${DATA_DIR}/${OUTPUT_PREFIX}-tranche-filtered-output.vcf.gz \
    --info-key CNN_1D \
    "${resources[@]}"

echo "👉 Done!!! See ${DATA_DIR}/${OUTPUT_PREFIX}-tranche-filtered-output.vcf.gz file."

date
