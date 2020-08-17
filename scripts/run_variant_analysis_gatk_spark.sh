#!/usr/bin/env bash

if [[ $# -ne 4 ]]; then
    echo "Usage: run_variant_analysis_gatk.sh <hs38|hs38a|hs38DH|hs37|hs37d5> <HDFS_PATH_OF_FASTQ_file1> <HDFS_PATH_OF_FASTQ_file2> <cluster size>"
    exit
fi

if [[ ! -f "${1}.fa" ]]; then
    echo "😡 Missing reference genome. Run setup_reference_genome.sh."
    exit
fi

SPARK_MASTER="spark://vm0:7077"
CANNOLI_SUBMIT=${CANNOLI_HOME}"/spark2_exec/cannoli-submit"
HDFS_PREFIX="hdfs://vm0:9000"
EXECUTOR_MEMORY=50g
DRIVER_MEMORY=50g
INPUT_FILE=mysequence
DATA_DIR="/mydata"
REFERENCE=${DATA_DIR}"/"${1}".fa"
DICT=${DATA_DIR}"/"${1}".dict"
GATK=${DATA_DIR}"/gatk-4.1.8.0/gatk"
OUTPUT_PREFIX="VA-"${USER}"-result"

let NUM_EXECUTORS=${3}
let NUM_CORES=$(nproc)-4

echo "👉 Deleting files..."
hdfs dfs -rm -r ${HDFS_PREFIX}/${INPUT_FILE}.*
rm -rvf ${HOME}/${OUTPUT_PREFIX}-gatk-spark-output.vcf

date
echo "👉 Interleaving FASTQ files."
${CANNOLI_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- interleaveFastq ${2} ${3} ${HDFS_PREFIX}/${INPUT_FILE}.ifq

echo "👉 Executing bwa for alignment."
${CANNOLI_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- bwa ${HDFS_PREFIX}/${INPUT_FILE}.ifq ${HDFS_PREFIX}/${INPUT_FILE}.bam \
    -executable ${BWA} -sample_id mysample -index ${REFERENCE} -sequence_dictionary ${DICT} -single -add_files

echo "👉 Sorting the bam file."
./gatk SortSamSpark -I ${HDFS_PREFIX}/${INPUT_FILE}.bam -O ${HDFS_PREFIX}/${INPUT_FILE}-sorted.bam \
    --spark-runner SPARK  --spark-master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY}

echo "👉 marking duplicates before variant calling."
./gatk MarkDuplicatesSpark -I ${HDFS_PREFIX}/${INPUT_FILE}-sorted.bam -O ${HDFS_PREFIX}/${OUTPUT_PREFIX}-rg-sorted-final.bam \
    --spark-runner SPARK  --spark-master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY}
    --tmp-dir ${DATA_DIR}/gatk-tmp

echo "👉 Running GATK HaplotypeCaller on spark for variant calling."
${GATK_HOME}/gatk HaplotypeCallerSpark \
    -R ${1}.fa \
    -I ${HDFS_PREFIX}/${OUTPUT_PREFIX}-rg-sorted-final.bam \
    -O ${HDFS_PREFIX}/${OUTPUT_PREFIX}-gatk-output.vcf \
    --spark-runner SPARK  --spark-master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY}

hdfs dfs -copyToLocal ${HDFS_PREFIX}/${OUTPUT_PREFIX}-gatk-output.vcf ${HOME}/${OUTPUT_PREFIX}-gatk-output.vcf
echo "👉 Done with variant calling. See ${OUTPUT_PREFIX}-gatk-output.vcf file."
date