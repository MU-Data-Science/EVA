#!/usr/bin/env bash
# Variant calling using GATK

DATA_DIR="/mydata"
BWA_HOME=${HOME}/bwa
SAMBAMBA_HOME=${HOME}
TMP_DIR="/mydata/tmp"
OUTPUT_PREFIX="VA-"${USER}"-result"
PICARD_JAR=${HOME}/picard.jar
GATK_HOME=${HOME}/gatk-4.1.8.0
SAMTOOLS_HOME=${HOME}/samtools
TRANCHE_RESOURCES=(\
  "${DATA_DIR}/hapmap_3.3.hg38.vcf.gz" \
  "${DATA_DIR}/1000G_omni2.5.hg38.vcf.gz" \
  "${DATA_DIR}/1000G_phase1.snps.high_confidence.hg38.vcf.gz")

echo "👉 Deleting files..."
rm -rvf ${DATA_DIR}/${OUTPUT_PREFIX}-*.vcf*

echo "👉 Validating the cluster."
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: run_variant_analysis_gatk.sh <hs38|hs38a|hs38DH|hs37|hs37d5> <FASTQ_file1> [FASTQ_file2]"
    exit
fi

if [[ ! -f "${1}.fa" ]]; then
    echo "😡 Missing reference genome. Run setup_reference_genome.sh."
    exit
fi

for file in "${TRANCHE_RESOURCES[@]}"; do
  if [[ (! -f "$file") || (! -f "${file}.tbi")]]; then
    echo "😡 Trance Resource ${file} or ${file}.tbi missing."
    exit
  fi
done

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
SAM2BAM_CMD="${SAMTOOLS_HOME}/samtools view -@${num_threads} -b ${OUTPUT_PREFIX}.sam.gz > ${OUTPUT_PREFIX}.bam"
eval ${SAM2BAM_CMD}
if [[ $? -eq 0 ]]; then
    echo "👉 Done with BAM conversion."
else
    echo "😡 Failed running BAM conversion."
    exit
fi

echo "👉 Adding Read Group to BAM file."

java -jar ${PICARD_JAR} AddOrReplaceReadGroups \
    I=${OUTPUT_PREFIX}.bam \
    O=${OUTPUT_PREFIX}-rg.bam \
    RGSM=mysample \
    RGPU=myunit \
    RGID=mygroupID \
    RGLB=mylib \
    RGPL=Illumina

echo "👉 Performing sorting of BAM file."

java -jar ${PICARD_JAR} SortSam \
    I=${OUTPUT_PREFIX}-rg.bam \
    O=${OUTPUT_PREFIX}-rg-sorted.bam \
    SORT_ORDER=coordinate

echo "👉 Marking duplicates in BAM file."
java -jar ${PICARD_JAR} MarkDuplicates \
    I=${OUTPUT_PREFIX}-rg-sorted.bam \
    O=${OUTPUT_PREFIX}-rg-sorted-final.bam \
    M=${OUTPUT_PREFIX}-rg-sorted-final-dup_metrics.txt

echo "👉 Index processed BAM file before variant calling."
${SAMTOOLS_HOME}/samtools index ${OUTPUT_PREFIX}-rg-sorted-final.bam

echo "👉 Running GATK HaplotypeCaller for variant calling."
${GATK_HOME}/gatk HaplotypeCaller -R ${1}.fa -I ${OUTPUT_PREFIX}-rg-sorted-final.bam -O ${OUTPUT_PREFIX}-gatk-output.vcf
echo "👉 Done with variant calling. See ${OUTPUT_PREFIX}-gatk-output.vcf file."

echo "👉 Running Base Quality Score Recalibration."
${HOME}/EVA/scripts/run_BQSR_single_node.sh ${1} ${OUTPUT_PREFIX}-gatk-output.vcf ${OUTPUT_PREFIX}

echo "👉 Filtering annotated variants using Convolutional Neural Net."
${GATK} CNNScoreVariants \
  -V ${DATA_DIR}/${OUTPUT_PREFIX}-output-gatk-spark-BQSR-output.vcf \
	-R ${REFERENCE} \
	-O ${DATA_DIR}/${OUTPUT_PREFIX}-cnn-annotated.vcf

echo "👉 Applying tranche filters"
for resource in "${TRANCHE_RESOURCES[@]}"; do
    resources+=( --resource "$resource" )
done

${GATK} FilterVariantTranches \
    -V ${DATA_DIR}/${OUTPUT_PREFIX}-cnn-annotated.vcf \
    -O ${DATA_DIR}/${OUTPUT_PREFIX}-tranche-filtered-output.vcf.gz \
    --info-key CNN_1D \
    "${resources[@]}"

echo "👉 Done!!! See ${DATA_DIR}/${OUTPUT_PREFIX}-tranche-filtered-output.vcf.gz file."

date
