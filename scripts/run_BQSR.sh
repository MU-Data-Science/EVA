#!/usr/bin/env bash
if [[ $# -ne 4 ]]; then
    echo "Usage: run_BQSR.sh <hs38|hs38a|hs38DH|hs37|hs37d5> <VCF file> <output_prefix> <cluster size>"
    exit
fi

SPARK_MASTER="spark://vm0:7077"
HDFS_PREFIX="hdfs://vm0:9000"
EXECUTOR_MEMORY=50g
DRIVER_MEMORY=50g
DATA_DIR="/mydata"
REFERENCE="file://"${DATA_DIR}"/"${1}".fa"
REF_CHECK=${DATA_DIR}"/"${1}."fa"
GATK=${DATA_DIR}"/gatk-4.1.8.0/gatk"

let NUM_EXECUTORS=${4}
let NUM_CORES=$(nproc)-4

echo "👉 Deleting files..."
hdfs dfs -rm -r ${HDFS_PREFIX}/${3}-output.bam

date

if [[ ! -f "${REF_CHECK}" ]]; then
    echo "😡 Missing reference genome. Run setup_reference_genome.sh."
    exit
fi

# We followed the steps provided here by Mohammed Khalfan:
# https://gencore.bio.nyu.edu/variant-calling-pipeline-gatk4/


${GATK} SelectVariants -R ${REFERENCE} \
    -V ${2} --select-type-to-include SNP -O ${3}-orig-snps.vcf

${GATK} SelectVariants -R ${REFERENCE} \
    -V ${2} --select-type-to-include INDEL -O ${3}-orig-indels.vcf

${GATK} VariantFiltration -R ${REFERENCE} -V ${3}-orig-snps.vcf \
    -O ${3}-filtered-snps.vcf \
    --filter-name "QD_filter" --filter-expression "QD < 2.0" \
    --filter-name "FS_filter" --filter-expression "FS > 60.0" \
    --filter-name "MQ_filter" --filter-expression "MQ < 40.0" \
    --filter-name "SOR_filter" --filter-expression "SOR > 4.0" \
    --filter-name "MQRankSum_filter" --filter-expression "MQRankSum < -12.5" \
    --filter-name "ReadPosRankSum_filter" --filter-expression "ReadPosRankSum < -8.0"

${GATK} VariantFiltration -R ${REFERENCE} -V ${3}-orig-indels.vcf \
    -O ${3}-filtered-indels.vcf \
    --filter-name "QD_filter" --filter-expression "QD < 2.0" \
    --filter-name "FS_filter" --filter-expression "FS > 200.0" \
    --filter-name "SOR_filter" --filter-expression "SOR > 10.0"

${GATK} SelectVariants --exclude-filtered \
        -V ${3}-filtered-snps.vcf \
        -O ${3}-BQSR-snps.vcf

${GATK} SelectVariants --exclude-filtered \
        -V ${3}-filtered-indels.vcf \
        -O ${3}-BQSR-indels.vcf

${GATK} BQSRPipelineSpark -R ${REFERENCE} \
    -I ${HDFS_PREFIX}/${3}-sorted.bam \
    --known-sites ${3}-BQSR-snps.vcf --known-sites ${3}-BQSR-indels.vcf \
    -O ${HDFS_PREFIX}/${3}-BQSR-output.bam \
    --spark-runner SPARK --spark-master ${SPARK_MASTER} \
    --conf "spark.executor.cores=${NUM_CORES}" --conf "spark.executor.memory=${EXECUTOR_MEMORY}" \
    --conf "spark.executor.instances=${NUM_EXECUTORS}"

${GATK} HaplotypeCallerSpark \
    -R ${REFERENCE} \
    -I ${HDFS_PREFIX}/${3}-BQSR-output.bam \
    -O ${HDFS_PREFIX}/${3}-output-gatk-spark-BQSR-output.vcf \
    --spark-runner SPARK --spark-master ${SPARK_MASTER} \
    --conf "spark.executor.cores=${NUM_CORES}" --conf "spark.executor.memory=${EXECUTOR_MEMORY}" \
    --conf "spark.executor.instances=${NUM_EXECUTORS}"

echo "👉 Completed the BQSR process."
date