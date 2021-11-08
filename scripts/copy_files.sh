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
	"hs38.fa.bwt.2bit.64" \
	"hs38.fa.0123" \
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

echo -e "Finished copying the files."
exit 0