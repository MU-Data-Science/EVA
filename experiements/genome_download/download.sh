#!/bin/bash

data_dir="$1"

if [ "$#" -ne 1 ]
then
  echo "Usage:   $0 <DATA_DIRECTORY>"
  exit 1
fi

for genome in $(cat Genome_List.txt)
do
  echo "Downloading contents from $genome directory."
  # Creating a directory for the sequence.
  mkdir -p $data_dir/$genome

  # Downloading the genome fragments.
  wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/$genome/sequence_read/*_1.filt.fastq.gz -P $data_dir/$genome
  wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/$genome/sequence_read/*_2.filt.fastq.gz -P $data_dir/$genome

  # Extracting the files.
  gunzip $data_dir/$genome/*.filt.fastq.gz
done

echo "Completed downloading all genome sequences."