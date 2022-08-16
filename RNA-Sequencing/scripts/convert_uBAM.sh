# Path: /mydata/EVA/scripts/convert_uBAM.sh
#!/usr/bin/env bash

# Configurations.
DATA_DIR="/mydata"
GATK_TEMP_DIR="${DATA_DIR}/gatk-tmp"
OUTPUT_PREFIX="${DATA_DIR}/VA-"${USER}"-result"
GATK=${DATA_DIR}"/gatk-4.1.8.0/gatk"

if [[ $# -ne 2 ]]; then
    echo "Usage: convert_uBAM.sh <PATH_TO_FASTQ_FILE_1> <PATH_TO_FASTQ_FILE_2>"
    exit
fi

echo "👉 Deleting previous execution outputs..."
rm -rvf ${OUTPUT_PREFIX}*

echo "👉 Converting fastq pairs to ubam"
${GATK} FastqToSam --java-options "-Djava.io.tmpdir=${GATK_TEMP_DIR}" \
  --FASTQ ${1} \
  --FASTQ2 ${2} \
  --SAMPLE_NAME RNASample \
  --OUTPUT ${OUTPUT_PREFIX}.unmapped.bam