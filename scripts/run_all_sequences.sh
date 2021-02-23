#!/usr/bin/env bash

EVA_HOME=${HOME}"/EVA"
HDFS_PREFIX="hdfs://vm0:9000"

if [[ $# -ne 2 ]]; then
    echo "Usage: run_all.sh <file> <cluster size>"
    exit
fi

# Delete *.ifq, *.bam*, *.vcf* files
$HADOOP_HOME/bin/hdfs dfs -rm -r /*.ifq /*.bam* /*.vcf*

cat ${1} | while read line;
do
    IFS=' '
    read -a sequence <<< "$line"
    file1=$HDFS_PREFIX"/"${sequence[1]}"_1.fastq.gz"
    file2=$HDFS_PREFIX"/"${sequence[1]}"_2.fastq.gz"
    echo $file1
    echo $file2
    echo "Starting variant analysis on "${sequence[1]}
    ${EVA_HOME}/scripts/run_variant_analysis_adam_basic.sh hs38 $file1 $file2 ${2} ${sequence[1]}
    echo "Completed variant analysis on "${sequence[1]}
done