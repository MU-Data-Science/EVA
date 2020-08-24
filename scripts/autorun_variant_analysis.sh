#!/bin/bash -i

# Constants
HS38_DIR="/proj/eva-public-PG0/Genome_Data"
DATA_DIR="/mydata"

if [[ $# -ne 4 ]]; then
    echo "Usage: autorun_variant_analysis.sh <hs38|hs38a|hs38DH|hs37|hs37d5> <FASTQ_FILE_1_URL> <FASTQ_FILE_2_URL> <CLUSTER_SIZE>"
    exit
fi

# Temporary command to setup hs38.
echo "Copying hs38 from the share directory"
cp -r ${HS38_DIR}/hs38* $DATA_DIR
#
#echo "Downloading the FASTQ files. (Presently only .gz extensions are supported.)"
wget $2 -O ${DATA_DIR}/Input_1.filt.fastq.gz
wget $3 -O ${DATA_DIR}/Input_2.filt.fastq.gz

echo "Uploading the files to HDFS."
hdfs dfs -copyFromLocal ${DATA_DIR}/Input_1.filt.fastq.gz /
hdfs dfs -copyFromLocal ${DATA_DIR}/Input_2.filt.fastq.gz /

echo "Performing variant analysis using adam."
${HOME}/EVA/scripts/run_variant_analysis_adam.sh hs38 hdfs://vm0:9000/Input_1.filt.fastq.gz hdfs://vm0:9000/Input_2.filt.fastq.gz $4