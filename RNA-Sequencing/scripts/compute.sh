# Path: /mydata/EVA/scripts/compute.sh

#!/usr/bin/env bash

# Defining the VCF directory.
VCF_DIR="/mydata/CADD_Inp"

# Defining the Output directory.
OUT_DIR="/mydata/CADD_Scores"

# Defining the CADD scripts directory.
CADD_DIR="/mydata/CADD-scripts"

# Iterating over the sample ID directories.
for file in ${VCF_DIR}/*
do
	# Obtaining the input VCF.
	echo "Computing CADD score for the input file ${file}"

    # Obtaining the out file name.
    filepath="${file%%.*}"
    filename=${filepath##*/}

    # Computing the CADD scores.
    cd ${CADD_DIR} && ./CADD.sh -g GRCh37 -o ${OUT_DIR}/${filename}.tsv.gz ${file} #For hg19

done