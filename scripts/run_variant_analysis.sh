#!/usr/bin/env bash

BWA_HOME=${HOME}/bwa
FREEBAYES_HOME=${HOME}/freebayes
TMP_DIR=/mydata/tmp

if [[ $# -ne 2 ]]; then
    echo "Usage: run_variant_analysis.sh <hs38|hs38a|hs37> <sequence_prefix>"
    echo "       (example of a sequence prefix: SRR062635)"
    exit
fi

if [[ ! -f "${1}.fa" ]]; then
    echo "😡 Missing reference genome. Run setup_reference_genome.sh."
    exit
fi

if [[ ! -f "${2}_1.filt.fastq.gz" || ! -f "${2}_2.filt.fastq.gz" ]]; then
    echo "😡 Missing files for sequence ${2}. Download from ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/."
    exit
fi

echo "👉 Starting alignment with bwa"
num_threads=$(nproc)
BWA_CMD="${BWA_HOME}/bwa mem -t ${num_threads} ${1}.fa ${2}_1.filt.fastq.gz ${2}_2.filt.fastq.gz | gzip > ${2}.sam.gz"
eval ${BWA_CMD}
if [[ $? -eq 0 ]]; then
    echo "👉 Done with alignment"
else
    echo "😡 Failed running bwa"
    exit
fi

echo "👉 Converting to BAM file"
SAM2BAM_CMD="samtools view -b ${2}.sam.gz > ${2}.bam"
eval ${SAM2BAM_CMD}
if [[ $? -eq 0 ]]; then
    echo "👉 Done with BAM conversion"
else
    echo "😡 Failed running BAM conversion"
    exit
fi

echo "👉 Performing sorting of BAM file"
rm -rf ${TMP_DIR}
mkdir ${TMP_DIR}
SORT_CMD="sambamba sort ${2}.bam --tmpdir=${TMP_DIR}"
eval ${SORT_CMD}
if [[ $? -eq 0 ]]; then
    echo "👉 Done with sorting BAM file"
else
    echo "😡 Failed sorting BAM file"
    exit
fi

echo "👉 Marking duplicates in BAM file"
MARKDUP_CMD="sambamba markdup ${2}.sorted.bam ${2}.final.bam --tmpdir=${TMP_DIR}"
eval ${MARKDUP_CMD}
if [[ $? -eq 0 ]]; then
    echo "👉 Done with marking duplicates in BAM file"
else
    echo "😡 Failed marking duplicates in BAM file"
    exit
fi

echo "👉 Running freebayes for variant calling"
FREEBAYES_CMD="${FREEBAYES_HOME}/bin/freebayes -f ${1}.fa ${2}.final.bam > ${2}.output.vcf"
eval ${FREEBAYES_CMD}
if [[ $? -eq 0 ]]; then
    echo "👉 Done with variant calling. See ${2}.output.vcf file."
else
    echo "😡 Failed performing variant calling"
    exit
fi

echo "👉 Done!"