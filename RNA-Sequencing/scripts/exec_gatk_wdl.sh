# Path: /mydata/gatk-workflows/exec_gatk_wdl.sh

#!/usr/bin/env bash

# Defining the workflow directory.
GATK_WDL_DIR="/mydata/gatk-workflows"

# Defining the workflow project name.
GATK_WDL="gatk4-rnaseq-germline-snps-indels"

# Defining directory containing the unmapped bam files.
VCF_DIR="/mydata/CADD_Inp"

# Iterating over the sample ID directories.
for dir in ${GATK_WDL_DIR}/inputs/*.unmapped.bam
do
	# Obtaining the unmapped bam.
	echo "Processing file ${dir}"

    # Adding the file to the template
    echo 'sed '"'"'46 i "RNAseq.inputBam": "'${dir}'"'"'"' '${GATK_WDL_DIR}'/'${GATK_WDL}'/template.json > '${GATK_WDL_DIR}'/'${GATK_WDL}'/gatk4-rna-germline-variant-calling.inputs.json' > temp.sh && bash temp.sh

    # Executing the workflow.
    cd ${GATK_WDL_DIR} && java -jar cromwell-33.1.jar run ./${GATK_WDL}/gatk4-rna-best-practices.wdl --inputs ./${GATK_WDL}/gatk4-rna-germline-variant-calling.inputs.json

	# Obtaining the VCF.
    sudo find ${GATK_WDL_DIR}/cromwell-executions/RNAseq -name '*variant_filtered.vcf.gz' -exec mv "{}" ${VCF_DIR}  \;

    # Cleaning up.
    cd ${GATK_WDL_DIR} && sudo rm -rvf cromwell-executions cromwell-workflow-logs

done
echo 'sed '"'"'46 i "RNAseq.inputBam": "'/mydata/gatk-workflows/inputs/ERR5429524.unmapped.bam'"'"'"' /mydata/gatk-workflows/gatk4-rnaseq-germline-snps-indels/template.json > /mydata/gatk-workflows/gatk4-rnaseq-germline-snps-indels/gatk4-rna-germline-variant-calling.inputs.json' > temp.sh && bash temp.sh

 cd /mydata/gatk-workflows && java -jar cromwell-33.1.jar run ./gatk4-rnaseq-germline-snps-indels/gatk4-rna-best-practices.wdl --inputs ./gatk4-rnaseq-germline-snps-indels/gatk4-rna-germline-variant-calling.inputs.json