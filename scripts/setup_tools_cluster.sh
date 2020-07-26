#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
  echo "Usage: copy_files <cluster size>"
  exit
fi

cd ${HOME}
${HOME}/EVA/scripts/setup_bwa_freebayes.sh

for ((i=1;i<${1};i++));
do
  ssh vm${i} 'cd ${HOME}; rm -rf EVA; git clone https://github.com/MU-Data-Science/EVA.git; ${HOME}/EVA/scripts/setup_bwa_freebayes.sh'
  echo "Completed installation on vm"${i}
done
