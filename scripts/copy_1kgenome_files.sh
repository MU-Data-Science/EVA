DEST_DIR="/mydata"
SOURCE_DIR="/proj/eva-public-PG0/Genome_Data/"
FILES=(
"1000G_omni2.5.hg38.vcf.gz"
"1000G_omni2.5.hg38.vcf.gz.tbi"
"1000G_phase1.snps.high_confidence.hg38.vcf.gz"
"1000G_phase1.snps.high_confidence.hg38.vcf.gz.tbi"
"hapmap_3.3.hg38.vcf.gz"
"hapmap_3.3.hg38.vcf.gz.tbi"
)

for file in "${FILES[@]}"; do
                echo $file
                cp $SOURCE_DIR$file $DEST_DIR
done
