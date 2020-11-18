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
rm -rvf ${HOME}/${OUTPUT_PREFIX}-gatk-spark-output.vcf

date
start=`date +%s`
echo "👉 Interleaving FASTQ files."
${CANNOLI_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- interleaveFastq ${2} ${3} ${HDFS_PREFIX}/${INPUT_FILE}.ifq
end_1=`date +%s`
runtime=$((end_1-start))
echo "Total time taken for Interleaving FASTQ files: " $runtime

if [[ ! -f "${REF_CHECK}" ]]; then
    echo "😡 Missing reference genome. Run setup_reference_genome.sh."
    exit
fi

echo "👉 Executing bwa for alignment."
${CANNOLI_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- bwaMem ${HDFS_PREFIX}/${INPUT_FILE}.ifq ${HDFS_PREFIX}/${INPUT_FILE}.bam \
    -executable ${BWA} -sample_id mysample -index ${REFERENCE} -sequence_dictionary ${DICT} -single -add_files
end_2=`date +%s`
runtime=$((end_2-end_1))
echo "Total time taken for Executing bwa for alignment: " $runtime

echo "👉 Sorting the bam file."
${GATK} SortSamSpark -I ${HDFS_PREFIX}/${INPUT_FILE}.bam -O ${HDFS_PREFIX}/${INPUT_FILE}-sorted.bam \
    --spark-runner SPARK --spark-master ${SPARK_MASTER} --conf "spark.executor.cores=${NUM_CORES}" --conf "spark.executor.memory=${EXECUTOR_MEMORY}" --conf "spark.executor.instances=${NUM_EXECUTORS}"
end_3=`date +%s`
runtime=$((end_3-end_2))
echo "Total time taken for Sorting the bam file: " $runtime

echo "👉 marking duplicates before variant calling."
${GATK} MarkDuplicatesSpark -I ${HDFS_PREFIX}/${INPUT_FILE}-sorted.bam -O ${HDFS_PREFIX}/${OUTPUT_PREFIX}-rg-sorted-final.bam \
    --spark-runner SPARK --spark-master ${SPARK_MASTER} --conf "spark.executor.cores=${NUM_CORES}" --conf "spark.executor.memory=${EXECUTOR_MEMORY}" --conf "spark.executor.instances=${NUM_EXECUTORS}" \
    --tmp-dir ${DATA_DIR}/gatk-tmp
end_4=`date +%s`
runtime=$((end_4-end_3))
echo "Total time taken for marking duplicates before variant calling: " $runtime

echo "👉 Running GATK HaplotypeCaller on spark for variant calling."
${GATK} HaplotypeCallerSpark \
    -R ${REFERENCE} \
    -I ${HDFS_PREFIX}/${OUTPUT_PREFIX}-rg-sorted-final.bam \
    -O ${HDFS_PREFIX}/${OUTPUT_PREFIX}-gatk-spark-output.vcf \
    --spark-runner SPARK --spark-master ${SPARK_MASTER} --conf "spark.executor.cores=${NUM_CORES}" --conf "spark.executor.memory=${EXECUTOR_MEMORY}" --conf "spark.executor.instances=${NUM_EXECUTORS}"
end_5=`date +%s`
runtime=$((end_5-end_4))
echo "Total time taken for Running GATK HaplotypeCaller on spark for variant calling: " $runtime

hdfs dfs -copyToLocal ${HDFS_PREFIX}/${OUTPUT_PREFIX}-gatk-spark-output.vcf ${HOME}/${OUTPUT_PREFIX}-gatk-spark-output.vcf
echo "👉 Done with variant calling. See ${OUTPUT_PREFIX}-gatk-spark-output.vcf file."

date
