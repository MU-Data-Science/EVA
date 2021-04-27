#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
    echo "Usage: convert_known_snps_indels_to_adam.sh <cluster size>"
    echo "       cluster_size - number of nodes in the cluster"
    exit
fi

SPARK_MASTER="spark://vm0:7077"
CANNOLI_SUBMIT=${CANNOLI_HOME}"/exec/cannoli-submit"
ADAM_SUBMIT=${ADAM_HOME}"/exec/adam-submit"
HDFS_PREFIX="hdfs://vm0:9000"
EXECUTOR_MEMORY=50g
DRIVER_MEMORY=50g
PROJ_DIR="/proj/eva-public-PG0"
KNOWN_SNPS_VCF=${PROJ_DIR}"/Homo_sapiens_assembly38.dbsnp138.vcf.gz"
KNOWN_INDELS_VCF=${PROJ_DIR}"/Homo_sapiens_assembly38.known_indels.vcf.gz"
KNOWN_SNPS_HDFS=${HDFS_PREFIX}"/known_snps"
KNOWN_INDELS_HDFS=${HDFS_PREFIX}"/known_indels"

DATA_DIR="/mydata"

let NUM_EXECUTORS=${1}
let NUM_CORES=$(nproc)-4

echo "👉 Validating the cluster."
if [[ ! -f "${KNOWN_SNPS_VCF}" || ! -f "${KNOWN_INDELS_VCF}" ]]; then
    echo "😡 Missing VCF files. Download them using wget. An example is shown below."
    echo "wget https://storage.googleapis.com/genomics-public-data/resources/broad/hg38/v0/Homo_sapiens_assembly38.dbsnp138.vcf"
    echo "bgzip Homo_sapiens_assembly38.dbsnp138.vcf"
    echo "wget https://storage.googleapis.com/genomics-public-data/resources/broad/hg38/v0/Homo_sapiens_assembly38.known_indels.vcf.gz"
    exit
fi

echo "👉 Deleting files..."
hdfs dfs -rm -r ${KNOWN_SNPS_HDFS} ${KNOWN_INDELS_HDFS}

date
echo "👉 Converting known SNPs to Adam format."
${ADAM_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- transformVariants "file:"${KNOWN_SNPS_VCF} ${KNOWN_SNPS_HDFS}

echo "👉 Converting known INDELs to Adam format."
${ADAM_SUBMIT} --master ${SPARK_MASTER} --driver-memory ${DRIVER_MEMORY} --num-executors ${NUM_EXECUTORS} --executor-cores ${NUM_CORES} --executor-memory ${EXECUTOR_MEMORY} \
    -- transformVariants "file:"${KNOWN_INDELS_VCF} ${KNOWN_INDELS_HDFS}

echo "👉 Done with conversion."

date
