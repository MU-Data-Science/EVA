#!/usr/bin/env bash

# Configurations.
DATA_DIR="/mydata"
OUTPUT_PREFIX="${DATA_DIR}/VA-"${USER}"-result"
RNASEQ_REF_FASTA="${DATA_DIR}/Homo_sapiens_assembly19_1000genomes_decoy.fasta"
RNASEQ_ANNOTATIONS_GTF="${DATA_DIR}/star.gencode.v19.transcripts.patched_contigs.gtf"
KNOWN_VCFS=(\
  "${DATA_DIR}/Homo_sapiens_assembly19_1000genomes_decoy.dbsnp138.vcf" \
  "${DATA_DIR}/Mills_and_1000G_gold_standard.indels.b37.sites.vcf" \
  "${DATA_DIR}/Homo_sapiens_assembly19_1000genomes_decoy.known_indels.vcf")
KNOWN_VCF_INDEXES=(\
  "${DATA_DIR}/Homo_sapiens_assembly19_1000genomes_decoy.dbsnp138.vcf.idx" \
  "${DATA_DIR}/Homo_sapiens_assembly19_1000genomes_decoy.known_indels.vcf.idx")

GATK=${DATA_DIR}"/gatk-4.1.8.0/gatk"
STAR="${DATA_DIR}/STAR/bin/Linux_x86_64/STAR"

if [[ $# -ne 2 ]]; then
    echo "Usage: run_RNAseq_short_variant_discovery.sh <PATH_TO_FASTQ_FILE_1> <PATH_TO_FASTQ_FILE_2>"
    exit
fi

echo "👉 Validating the Reference files."
if [[ (! -f "${RNASEQ_REF_FASTA}") || (! -f "${RNASEQ_ANNOTATIONS_GTF}")]]; then
    echo "😡 References ${RNASEQ_REF_FASTA} or ${RNASEQ_ANNOTATIONS_GTF} missing."
    exit
fi

for file in "${KNOWN_VCFS[@]}"; do
  if [[ (! -f "$file") ]]; then
    echo "😡 Known vcf ${file} missing."
    exit
  fi
done

for file in "${KNOWN_VCF_INDEXES[@]}"; do
  if [[ (! -f "$file") ]]; then
    echo "😡 Known vcf index ${file} missing."
    exit
  fi
done

let num_threads=$(nproc)

echo "👉 Converting fastq pairs to ubam"
${GATK} FastqToSam \
  --FASTQ ${1} \
  --FASTQ2 ${2} \
  --SAMPLE_NAME RNASample \
  --OUTPUT ${OUTPUT_PREFIX}.unmapped.bam

echo "👉 Mapping to Reference using STAR"

if [ -d "${DATA_DIR}/STAR2_5" ]; then
  echo "Creating STAR References"
  ${STAR} \
    --runMode genomeGenerate \
    --genomeDir ${DATA_DIR}/STAR2_5 \
    --genomeFastaFiles ${RNASEQ_REF_FASTA} \
    --sjdbGTFfile ${RNASEQ_ANNOTATIONS_GTF} \
    --runThreadN ${num_threads}
fi

${STAR} \
  --genomeDir ${DATA_DIR}/STAR2_5 \
  --runThreadN ${num_threads} \
  --readFilesIn ${1} ${2} \
  --readFilesCommand "gunzip -c" \
  --twopassMode Basic \
  --outFileNamePrefix ${OUTPUT_PREFIX}.star.

echo "👉 Merging alignment data from the STAR SAM with data in the unmapped BAM file."
${GATK} MergeBamAlignment \
  --REFERENCE_SEQUENCE ${RNASEQ_REF_FASTA} \
  --UNMAPPED_BAM ${OUTPUT_PREFIX}.unmapped.bam \
  --ALIGNED_BAM ${OUTPUT_PREFIX}.star.Aligned.out.sam \
  --OUTPUT ${OUTPUT_PREFIX}-merged.bam \
  --INCLUDE_SECONDARY_ALIGNMENTS false

echo "👉 Identifying duplicate reads."
${GATK} MarkDuplicates \
  --INPUT ${OUTPUT_PREFIX}-merged.bam \
  --OUTPUT ${OUTPUT_PREFIX}-MarkDup.bam  \
  --CREATE_INDEX true \
  --VALIDATION_STRINGENCY SILENT \
  --METRICS_FILE ${OUTPUT_PREFIX}.dedupped.metrics

echo "👉 Splitting Reads with N in Cigar"
${GATK} SplitNCigarReads \
  -R ${RNASEQ_REF_FASTA} \
  -I ${OUTPUT_PREFIX}-MarkDup.bam \
  -O ${OUTPUT_PREFIX}-SplitNCigarReads.bam

echo "Generating recalibration table for Base Quality Score Recalibration (BQSR)"
for vcf in "${KNOWN_VCFS[@]}"; do
    knownSites+=( --known-sites "${vcf}" )
done

${GATK} BaseRecalibrator \
  -R ${RNASEQ_REF_FASTA} \
  -I ${OUTPUT_PREFIX}-SplitNCigarReads.bam \
  --use-original-qualities \
  -O ${OUTPUT_PREFIX}.recal_data.csv \
  "${knownSites[@]}"

# Reference File URL's.
# gs://gatk-test-data/intervals/star.gencode.v19.transcripts.patched_contigs.gtf
# gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.fasta
# gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.fasta.fai
# gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.dict
# gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.dbsnp138.vcf
# gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.dbsnp138.vcf.idx
# gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Mills_and_1000G_gold_standard.indels.b37.sites.vcf
# gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.known_indels.vcf
# gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Mills_and_1000G_gold_standard.indels.b37.sites.vcf.idx
# gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.known_indels.vcf.idx
# gs://gatk-test-data/intervals/star.gencode.v19.transcripts.patched_contigs.gtf