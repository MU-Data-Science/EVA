#!/usr/bin/env bash

DATA_DIR="/mydata"
FILES_DIR="/proj/eva-public-PG0/Genome_Data"
NODE_PREFIX="vm"
FILES=(\
	"hs38.dict" \
	"hs38.fa" \
	"hs38.fa.amb" \
	"hs38.fa.ann" \
	"hs38.fa.bwt" \
	"hs38.fa.fai" \
	"hs38.fa.pac" \
	"hs38.fa.sa" \
	"hs38.fa.img" \
	)

FILESBQSR=(\
	"../Homo_sapiens_assembly38.dbsnp138.vcf.gz" \
	"../Homo_sapiens_assembly38.dbsnp138.vcf.gz.tbi" \
	"../Homo_sapiens_assembly38.known_indels.vcf.gz" \
	"../Homo_sapiens_assembly38.known_indels.vcf.gz.tbi" \
  )

if [[ $# -ne 1 ]]; then
	echo "usage: copy_files.sh <NO_OF_NODES>"
    exit
fi

p_list=""

for ((i=0;i<${1};i++)); do
	for file in "${FILES[@]}"; do
		ssh ${NODE_PREFIX}${i} "cp ${FILES_DIR}/${file} ${DATA_DIR}" &
		p_list="${p_list} $!"
	done
done

echo -e "Waiting for copying to finish."
finished=1
while [[ ${finished} -gt 0 ]]; do
	finished=${1}

  states=""
	for pid in ${p_list}; do
		state=$(ps -o state ${pid}  |tail -n +2)
		states="${states} ${state}"
		if [[ ${#state} -eq 0 ]]; then
			finished=$((finished-1))
		fi;
	done;

	states=${states// /}
	if [[ ${#states} -gt 0 ]]; then
		sleep 30
	fi
done;
wait

echo -e "Finished copying the reference files."

p_list=""

for ((i=0;i<${1};i++)); do
	for file in "${FILESBQSR[@]}"; do
		ssh ${NODE_PREFIX}${i} "cp ${FILES_DIR}/${file} ${DATA_DIR}" &
		p_list="${p_list} $!"
	done
done

echo -e "Waiting for copying to finish."
finished=1
while [[ ${finished} -gt 0 ]]; do
	finished=${1}

  states=""
	for pid in ${p_list}; do
		state=$(ps -o state ${pid}  |tail -n +2)
		states="${states} ${state}"
		if [[ ${#state} -eq 0 ]]; then
			finished=$((finished-1))
		fi;
	done;

	states=${states// /}
	if [[ ${#states} -gt 0 ]]; then
		sleep 30
	fi
done;
wait
echo -e "Finished copying known SNPs/INDELs files."

exit 0