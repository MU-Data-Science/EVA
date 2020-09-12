#!/usr/bin/env bash

CANNOLI_HOME_DIR="/mydata/cannoli"
SPARK_HOME_DIR="/mydata/spark"
HDFS_PREFIX="hdfs://vm0:9000"
LOCAL_PREFIX="file://"
MASTER_URL="yarn --deploy-mode client"
DATE=$(date "+%Y-%m-%d-%s")
LOGFILE="/mydata/${USER}-denovo-${DATE}.log"
EVA_JAR=${HOME}"/EVA/lib/eva-denovo_2.12-0.1.jar"

if [[ $# -ne 3 ]]; then
    echo "Usage: run_denovo.sh <file containing sample IDs> <file containing FASTQ URLs> <cluster size>"
    exit
fi

let NUM_EXECUTORS=${3}

$SPARK_HOME/bin/spark-submit --master ${MASTER_URL} --num-executors ${NUM_EXECUTORS} \
    --conf spark.yarn.appMasterEnv.CANNOLI_HOME=${CANNOLI_HOME_DIR} \
    --conf spark.yarn.appMasterEnv.SPARK_HOME=${SPARK_HOME_DIR} \
    --conf spark.executorEnv.CANNOLI_HOME=${CANNOLI_HOME_DIR} \
    --conf spark.executorEnv.SPARK_HOME=${SPARK_HOME_DIR} \
    ${EVA_JAR} -i ${LOCAL_PREFIX}/${1} -d ${LOCAL_PREFIX}/${2} &> ${LOGFILE} &

echo "See log file for progress: "${LOGFILE}