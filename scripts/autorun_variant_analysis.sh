#!/bin/bash -i

# Constants
HS38_DIR="/proj/eva-public-PG0/Genome_Data"
DATA_DIR="/mydata"
EXTENSIONS=(\
  ".fa" \
  ".fa.amb" \
  ".fa.ann" \
  ".fa.bwt" \
  ".fa.fai" \
  ".fa.pac" \
  ".fa.sa" \
  ".dict")

# Input Validation
if [[ $# -ne 4 ]]; then
    echo "Usage: autorun_variant_analysis.sh <hs38|hs38a|hs38DH|hs37|hs37d5> <FASTQ_FILE_1_URL> <FASTQ_FILE_2_URL> <CLUSTER_SIZE>"
    exit
fi

echo "👉 Clearning up data from previous executions."
rm -rvf ${DATA_DIR}/Input_1.filt.fastq.gz ${DATA_DIR}/Input_2.filt.fastq.gz
hdfs dfs -rm /Input_1.filt.fastq.gz
hdfs dfs -rm /Input_2.filt.fastq.gz

echo "👉 Verifying if reference genome is present."
for extension in "${EXTENSIONS[@]}"; do
  if [[ ! -f "${DATA_DIR}/${1}${extension}" ]]; then
    echo "😡 Missing reference genome. Executing setup_reference_genome.sh."
    cd ${DATA_DIR} && ${HOME}/EVA/scripts/setup_reference_genome.sh ${1}
    break
  fi
done

echo "👉 Downloading the FASTQ files. (Presently only .gz extensions are supported.)"
wget ${2} -O ${DATA_DIR}/Input_1.filt.fastq.gz
wget ${3} -O ${DATA_DIR}/Input_2.filt.fastq.gz

echo "👉 Uploading the files to HDFS."
hdfs dfs -copyFromLocal ${DATA_DIR}/Input_1.filt.fastq.gz /
hdfs dfs -copyFromLocal ${DATA_DIR}/Input_2.filt.fastq.gz /

echo "👉 Performing variant analysis using adam."
${HOME}/EVA/scripts/run_variant_analysis_adam.sh ${1} hdfs://vm0:9000/Input_1.filt.fastq.gz hdfs://vm0:9000/Input_2.filt.fastq.gz ${4}