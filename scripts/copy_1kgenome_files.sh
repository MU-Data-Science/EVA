DEST_DIR="/mydata"
SOURCE_DIR="/proj/eva-public-PG0/Genome_Data/"
FILES=(
"1000G_omni2.5.hg38.vcf.gz"
"1000G_omni2.5.hg38.vcf.gz.tbi"
"1000G_phase1.snps.high_confidence.hg38.vcf.gz"
"1000G_phase1.snps.high_confidence.hg38.vcf.gz.tbi"
"hapmap_3.3.hg38.vcf.gz"
"hapmap_3.3.hg38.vcf.gz.tbi"
"hs38.dict"
"hs38.fa"
"hs38.fa.ann"
"hs38.fa.img"  
"hs38.fa.sa"
"hs38.fa.amb"
"hs38.fa.bwt"
"hs38.fa.fai"
"hs38.fa.pac"
)

for file in "${FILES[@]}"; do
                echo $file
                cp $SOURCE_DIR$file $DEST_DIR
done
