#!/usr/bin/env bash

if [[ $# -ne 4 ]]; then
    echo "Usage: run_variant_analysis_gatk_spark.sh <hs38|hs38a|hs38DH|hs37|hs37d5> <HDFS_PATH_OF_FASTQ_file1> <HDFS_PATH_OF_FASTQ_file2> <cluster size>"
    exit
fi

SPARK_MASTER="spark://vm0:7077"
CANNOLI_SUBMIT=${CANNOLI_HOME}"/spark2_exec/cannoli-submit"
HDFS_PREFIX="hdfs://vm0:9000"
EXECUTOR_MEMORY=50g
DRIVER_MEMORY=50g
INPUT_FILE=mysequence
DATA_DIR="/mydata"
REFERENCE="file://"${DATA_DIR}"/"${1}".fa"
REF_CHECK=${DATA_DIR}"/"${1}."fa"
DICT="file://"${DATA_DIR}"/"${1}".dict"
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

echo "👉 Sorting the bam file."
${GATK} SortSamSpark -I ${HDFS_PREFIX}/${INPUT_FILE}.bam -O ${HDFS_PREFIX}/${INPUT_FILE}-sorted.bam \
    --spark-runner SPARK --spark-master ${SPARK_MASTER} --conf "spark.executor.cores=${NUM_CORES}" --conf "spark.executor.memory=${EXECUTOR_MEMORY}" --conf "spark.executor.instances=${NUM_EXECUTORS}"

echo "👉 marking duplicates before variant calling."
${GATK} MarkDuplicatesSpark -I ${HDFS_PREFIX}/${INPUT_FILE}-sorted.bam -O ${HDFS_PREFIX}/${OUTPUT_PREFIX}-rg-sorted-final.bam \
    --spark-runner SPARK --spark-master ${SPARK_MASTER} --conf "spark.executor.cores=${NUM_CORES}" --conf "spark.executor.memory=${EXECUTOR_MEMORY}" --conf "spark.executor.instances=${NUM_EXECUTORS}" \
    --tmp-dir ${DATA_DIR}/gatk-tmp

echo "👉 Running GATK HaplotypeCaller on spark for variant calling."
${GATK} HaplotypeCallerSpark \
    -R ${REFERENCE} \
    -I ${HDFS_PREFIX}/${OUTPUT_PREFIX}-rg-sorted-final.bam \
    -O ${HDFS_PREFIX}/${OUTPUT_PREFIX}-gatk-spark-output.vcf \
    --spark-runner SPARK --spark-master ${SPARK_MASTER} --conf "spark.executor.cores=${NUM_CORES}" --conf "spark.executor.memory=${EXECUTOR_MEMORY}" --conf "spark.executor.instances=${NUM_EXECUTORS}"

hdfs dfs -copyToLocal ${HDFS_PREFIX}/${OUTPUT_PREFIX}-gatk-spark-output.vcf ${DATA_DIR}/${OUTPUT_PREFIX}-gatk-spark-output.vcf
echo "👉 Done with variant calling. See ${OUTPUT_PREFIX}-gatk-spark-output.vcf file."

echo "👉 Running Base Quality Score Recalibration."
${HOME}/EVA/scripts/run_BQSR.sh ${1} ${DATA_DIR}/${OUTPUT_PREFIX}-gatk-spark-output.vcf ${OUTPUT_PREFIX} ${4}

echo "👉 Filtering annotated variants using Convolutional Neural Net."
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
