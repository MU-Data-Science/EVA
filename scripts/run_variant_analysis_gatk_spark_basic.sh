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

let NUM_EXECUTORS=${4}
let NUM_CORES=$(nproc)-4

echo "👉 Deleting files..."
hdfs dfs -rm -r ${HDFS_PREFIX}/${INPUT_FILE}*
hdfs dfs -rm -r ${HDFS_PREFIX}/${OUTPUT_PREFIX}*
rm -rvf ${DATA_DIR}/${OUTPUT_PREFIX}-*.vcf*
rm -rf ${DATA_DIR}/temp.bam
rm -rf ${DATA_DIR}/temp_?.fastq.gz

echo "👉 Validating the cluster."
if [[ ! -f "${REF_CHECK}" ]]; then
    echo "😡 Missing reference genome. Run setup_reference_genome.sh."
    exit
fi

date

echo "👉 Copying HDFS files to local storage"
hdfs dfs -get ${2} ${DATA_DIR}/temp_1.fastq.gz
hdfs dfs -get ${3} ${DATA_DIR}/temp_2.fastq.gz

echo "👉 Create the .bam file."
${GATK} FastqToSam -F1 ${DATA_DIR}/temp_1.fastq.gz -F2 ${DATA_DIR}/temp_1.fastq.gz -O ${DATA_DIR}/temp.bam --SAMPLE_NAME mysample

echo "👉 Copying the .bam file to HDFS."
hdfs dfs -put ${DATA_DIR}/temp.bam /${INPUT_FILE}_unaligned.bam

echo "👉 Using BWA/Mark Duplicates pipeline."
${GATK} BwaAndMarkDuplicatesPipelineSpark -I ${HDFS_PREFIX}/${INPUT_FILE}_unaligned.bam -O ${HDFS_PREFIX}/${INPUT_FILE}-final.bam -R ${REFERENCE} \
    -- --spark-runner SPARK --spark-master ${SPARK_MASTER} --conf "spark.executor.cores=${NUM_CORES}" --conf "spark.executor.memory=${EXECUTOR_MEMORY}" --conf "spark.executor.instances=${NUM_EXECUTORS}"

echo "👉 Sorting before variant calling."
${GATK} SortSamSpark -I ${HDFS_PREFIX}/${INPUT_FILE}-final.bam -O ${HDFS_PREFIX}/${INPUT_FILE}-final-sorted.bam \
    -- --spark-runner SPARK --spark-master ${SPARK_MASTER} --conf "spark.executor.cores=${NUM_CORES}" --conf "spark.executor.memory=${EXECUTOR_MEMORY}" --conf "spark.executor.instances=${NUM_EXECUTORS}"

echo "👉 Running GATK HaplotypeCaller on Spark for variant calling."
${GATK} HaplotypeCallerSpark \
    -R ${REFERENCE} \
    -I ${HDFS_PREFIX}/${INPUT_FILE}-final-sorted.bam \
    -O ${HDFS_PREFIX}/${OUTPUT_PREFIX}-gatk-spark-output.vcf \
    -- --spark-runner SPARK --spark-master ${SPARK_MASTER} --conf "spark.executor.cores=${NUM_CORES}" --conf "spark.executor.memory=${EXECUTOR_MEMORY}" --conf "spark.executor.instances=${NUM_EXECUTORS}"

hdfs dfs -get ${HDFS_PREFIX}/${OUTPUT_PREFIX}-gatk-spark-output.vcf ${DATA_DIR}/${OUTPUT_PREFIX}-gatk-spark-output.vcf
echo "👉 Done with variant calling. See ${OUTPUT_PREFIX}-gatk-spark-output.vcf file."

date