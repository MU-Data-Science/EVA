# Developed by Arun George Zacharaiah. Last updated by Shivika Prasanna on Aug 11, 2022.
# Execution steps for RNA-Sequencing.

#!/usr/bin/env bash

old="$IFS"
IFS=','
SEQ_ID_LIST_STR=""$*""
echo "$SEQ_ID_LIST_STR"
IFS=$old

DOWNLOAD_DIR="/mydata/InpSequences"
BAM_DIR="/mydata/Out_uBAM"
GATK_WORKFLOW_DIR="/mydata/gatk-workflows"
VCF_DIR="/mydata/CADD_Inp"
CADD_SCORES_DIR="/mydata/CADD_Scores"
CADD_SCRIPTS_DIR="/mydata/CADD-scripts"
EVA_HOME="/mydata/EVA"

EMAIL="spn8y@umsystem.edu"

echo "Cleaning up."
rm -rvf ${DOWNLOAD_DIR}/*
rm -rvf ${BAM_DIR}/*
rm -rvf ${GATK_WORKFLOW_DIR}/inputs/*.unmapped.bam
rm -rvf ${VCF_DIR}/*
rm -rvf ${CADD_SCORES_DIR}/*

echo "Downloading the sequences."
cd ${DOWNLOAD_DIR} && curl -X POST -H "Content-Type: application/x-www-form-urlencoded" -d "result=read_run&includeAccessions=${SEQ_ID_LIST_STR}&field=fastq_ftp&includeAccessionType=experiment" "https://www.ebi.ac.uk/ena/portal/api/files" --output ena_files.zip

echo "Extracting the sequences."
cd ${DOWNLOAD_DIR} && unzip ena_files.zip

echo "Converting .fastq to .unmapped.bam."
cd ${EVA_HOME}/scripts && bash recursive_convert_uBAM.sh

echo "Moving .unmapped.bam file to the GATK workflow directory."
mv ${BAM_DIR}/*.unmapped.bam ${GATK_WORKFLOW_DIR}/inputs

echo "Executing the GATK Workflows."
cd ${GATK_WORKFLOW_DIR} && bash exec_gatk_wdl.sh

echo "Computing CADD scores."
cd ${CADD_SCRIPTS_DIR} && bash compute.sh

echo "Completed processing. Sending out an email notification."
mail -s "Completed process notification" ${EMAIL} <<< "VCF and CADD Scores have been computed for ${SEQ_ID_LIST_STR}."
    