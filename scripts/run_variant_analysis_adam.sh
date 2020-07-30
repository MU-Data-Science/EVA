#!/usr/bin/env bash


if [[ $# -ne 4 ]]; then
    echo "Usage: run_variant_analysis_adam.sh <HDFS_PATH_OF_FASTQ_file1> <HDFS_PATH_OF_FASTQ_file2> <cluster size> <data directory>"
    exit
fi

SPARK_MASTER="spark://vm0:7077"
CANNOLI_SUBMIT=${EVA_HOME}"/cannoli_bin/cannoli-submit"
ADAM_SUBMIT=${EVA_HOME}"/adam_bin/adam-submit"
ADAM_SHELL=${EVA_HOME}"/adam_bin/adam-shell"
HDFS_PREFIX="hdfs://vm0:9000"
EXECUTOR_MEMORY=50g
DRIVER_MEMORY=50g
INPUT_FILE=mysequence
REFERENCE="/mydata/hs38.fa"
DICT="/mydata/hs38.dict"
DATA_DIR=${4}
FREE_BAYES=${DATA_DIR}"/freebayes/bin/freebayes"
BWA=${DATA_DIR}"/bwa/bwa"
OUTPUT_PREFIX="VA-"${USER}"-result"

let NUM_EXECUTORS=${3}
let NUM_CORES=$(nproc)-4

# Cleanup
echo "👉 Deleting files..."
hdfs dfs -rm -r ${HDFS_PREFIX}/${INPUT_FILE}.*

date
echo "👉 Interleaving FASTQ files."
${CANNOLI_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- interleaveFastq ${1} ${2} ${HDFS_PREFIX}/${INPUT_FILE}.ifq

echo "👉 Executing bwa for alignment."
${CANNOLI_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- bwa ${HDFS_PREFIX}/${INPUT_FILE}.ifq ${HDFS_PREFIX}/${INPUT_FILE}.bam \
    -executable ${BWA} -sample_id mysample -index ${REFERENCE} -sequence_dictionary ${DICT} -single -add_files

echo "👉 Sorting and marking duplicates before variant calling."
${ADAM_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- transformAlignments ${HDFS_PREFIX}/${INPUT_FILE}.bam ${HDFS_PREFIX}/${INPUT_FILE}.bam.adam \
    -mark_duplicate_reads -sort_by_reference_position_and_index

echo "👉 Variant calling using freebayes."
${CANNOLI_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- freebayes ${HDFS_PREFIX}/${INPUT_FILE}.bam.adam ${HDFS_PREFIX}/${INPUT_FILE}.variants.adam \
    -executable ${FREE_BAYES} -reference ${REFERENCE} -add_files

VCF_CMD1="import org.bdgenomics.adam.rdd.ADAMContext._ \n" 
VCF_CMD2="var variants = sc.loadParquetVariantContexts(\""${HDFS_PREFIX}"/"${INPUT_FILE}".variants.adam\") \n"
VCF_CMD3="variants.saveAsVcf(\""${HDFS_PREFIX}"/"${INPUT_FILE}".vcf\")"

echo -e ${VCF_CMD1}${VCF_CMD2}${VCF_CMD3} | ${ADAM_SHELL} --master ${SPARK_MASTER}
hdfs dfs -copyToLocal ${HDFS_PREFIX}/${INPUT_FILE}.vcf ${HOME}/${OUTPUT_PREFIX}-fbayes-output.vcf
echo "👉 Done with variant analysis. See ${HOME}/${OUTPUT_PREFIX}-fbayes-output.vcf."
date