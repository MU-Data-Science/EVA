#!/usr/bin/env bash

BWA_HOME=${HOME}/bwa

if [[ $# -ne 1 ]]; then
    echo "Usage: setup_reference_genome.sh <hs38|hs38a|hs37>"
    exit
fi

echo "Current working directory: "${PWD}

echo "👉 Starting to download the reference genome 😎"
$BWA_HOME/bwakit/run-gen-ref ${1}
echo "👉 Done with downloading the reference genome 😎"

echo "👉 Starting to index the reference genome 😎"
$BWA_HOME/bwa index ${1}.fa
samtools faidx ${1}.fa
echo "👉 Done with indexing the reference genome 😎"

echo "👉 Starting to create dictionary the reference genome 😎"
samtools dict -o ${1}.dict ${1}.fa
echo "👉 Done with creating the dictionary 😎"

echo "👉 Done!"