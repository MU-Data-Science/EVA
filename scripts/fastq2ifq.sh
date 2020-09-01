#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
    echo "Usage: fastq2ifq.sh <Sample ID>"
    echo "    "
    echo " The FASTQ files for Sample ID are assumed to be in HDFS."
    exit
fi

SPARK_MASTER="spark://vm0:7077"
CANNOLI_SUBMIT=${CANNOLI_HOME}"/exec/cannoli-submit"
HDFS_PREFIX="hdfs://vm0:9000"

echo "👉 Begin interleaving FASTQ files."
${CANNOLI_SUBMIT} --master ${SPARK_MASTER} -- interleaveFastq \
    ${HDFS_PREFIX}/${1}"_1.filt.fastq.gz" ${HDFS_PREFIX}/${1}"_2.filt.fastq.gz" ${HDFS_PREFIX}/${1}.ifq.gz

echo "👉 Completed interleaving FASTQ files."