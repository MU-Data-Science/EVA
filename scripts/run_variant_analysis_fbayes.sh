#!/usr/bin/env bash
# Variant calling using freebayes

BWA_HOME=${HOME}/bwa
SAMBAMBA_HOME=${HOME}
FREEBAYES_HOME=${HOME}/freebayes
TMP_DIR="/mydata/tmp"
OUTPUT_PREFIX="VA-"${USER}"-result"

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: run_variant_analysis.sh <hs38|hs38a|hs38DH|hs37|hs37d5> <FASTQ_file1> [FASTQ_file2]"
    exit
fi

if [[ ! -f "${1}.fa" ]]; then
    echo "😡 Missing reference genome. Run setup_reference_genome.sh."
    exit
fi

echo "👉 Starting alignment with bwa."
num_threads=$(nproc)
if [[ $# -eq 2 ]]; then
    BWA_CMD="${BWA_HOME}/bwa mem -t ${num_threads} ${1}.fa ${2} | gzip > ${OUTPUT_PREFIX}.sam.gz"
    if [[ ! -f "${2}" ]]; then
        echo "😡 Missing FASTQ input file. Cannot run bwa."
        exit
    fi
elif [[ $# -eq 3 ]]; then
   if [[ ! -f "${2}" || ! -f "${3}" ]]; then
        echo "😡 Missing FASTQ input files. Cannot run bwa."
        exit
    fi
    BWA_CMD="${BWA_HOME}/bwa mem -t ${num_threads} ${1}.fa ${2} ${3} | gzip > ${OUTPUT_PREFIX}.sam.gz"
else
    echo "😡 Something is wrong..."
    exit
fi
eval ${BWA_CMD}
if [[ $? -eq 0 ]]; then
    echo "👉 Done with alignment."
else
    echo "😡 Failed running bwa."
    exit
fi

echo "👉 Converting to BAM file."
SAM2BAM_CMD="samtools view -@${num_threads} -b ${OUTPUT_PREFIX}.sam.gz > ${OUTPUT_PREFIX}.bam"
eval ${SAM2BAM_CMD}
if [[ $? -eq 0 ]]; then
    echo "👉 Done with BAM conversion."
else
    echo "😡 Failed running BAM conversion."
    exit
fi

echo "👉 Performing sorting of BAM file."
rm -rf ${TMP_DIR}
mkdir ${TMP_DIR}
SORT_CMD="${SAMBAMBA_HOME}/sambamba sort -t ${num_threads} ${OUTPUT_PREFIX}.bam --tmpdir=${TMP_DIR}"
eval ${SORT_CMD}
if [[ $? -eq 0 ]]; then
    echo "👉 Done with sorting BAM file."
else
    echo "😡 Failed sorting BAM file."
    exit
fi

echo "👉 Marking duplicates in BAM file."
MARKDUP_CMD="${SAMBAMBA_HOME}/sambamba markdup -t ${num_threads} ${OUTPUT_PREFIX}.sorted.bam ${OUTPUT_PREFIX}.final.bam --tmpdir=${TMP_DIR}"
eval ${MARKDUP_CMD}
if [[ $? -eq 0 ]]; then
    echo "👉 Done with marking duplicates in BAM file."
else
    echo "😡 Failed marking duplicates in BAM file."
    exit
fi

echo "👉 Running freebayes for variant calling."
FREEBAYES_CMD="${FREEBAYES_HOME}/bin/freebayes -f ${1}.fa ${OUTPUT_PREFIX}.final.bam > ${OUTPUT_PREFIX}-fbayes.output.vcf"
eval ${FREEBAYES_CMD}
if [[ $? -eq 0 ]]; then
    echo "👉 Done with variant calling. See ${OUTPUT_PREFIX}-fbayes.output.vcf file."
else
    echo "😡 Failed performing variant calling."
    exit
fi

echo "👉 Done!"