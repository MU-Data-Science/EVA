# Path: /mydata/EVA/scripts/recursive_convert_uBAM.sh
#!/usr/bin/env bash

# Pre-requisites:
# 1. Create the directory InpSequences (/mydata/InpSequences).
# 2. Create the directory Out_uBAM (/mydata/Out_uBAM).
# 3. Place the directory containing the sample ids in the format /mydata/InpSequences/<ID>/<ID>.gz.
# 4. Place the script in EVA/scripts.
# 5. Give exec permission to convert_uBAM.sh (chmod +x /mydata/EVA/scripts/convert_uBAM.sh)

# Defining the directory containing the IDs.
DATA_DIR="/mydata/InpSequences"

# Defining the directory to hold Output VCFs
OUT_uBAM="/mydata/Out_uBAM"

# Iterating over the sample ID directories.
for dir in ${DATA_DIR}/*
do
	# Obtaining the sample IDs
	echo "Entering ${dir}"
	SAMPLE_ID=${dir##*/}
	echo "Processing sample ${SAMPLE_ID}."

	# Creating the fastq split paths.
	FASTQ_FILE_1="${DATA_DIR}/${SAMPLE_ID}/${SAMPLE_ID}_1.fastq.gz"
	FASTQ_FILE_2="${DATA_DIR}/${SAMPLE_ID}/${SAMPLE_ID}_2.fastq.gz"

	# Converting fastq splits to unmapped BAM.
	./convert_uBAM.sh ${FASTQ_FILE_1} ${FASTQ_FILE_2}

	# Moving the output VCF to the output directory.
	mv /mydata/VA-${USER}-result.unmapped.bam ${OUT_uBAM}/${SAMPLE_ID}.unmapped.bam

	echo "Completed processing ${SAMPLE_ID}."
done

echo "Completed processing all sequences."