#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
  echo "Usage: copy_files <cluster size>"
  exit
fi

# First copy to vm0
cp -R  /proj/sparksecurity-PG0/hs38/* /mydata/
cd ${HOME}
git clone -b development https://github.com/MU-Data-Science/EVA.git
${HOME}/EVA/scripts/setup_bwa_freebayes.sh

# Then to the rest of the nodes
for ((i=1;i<${1};i++));
do
  ssh vm${i} 'cd ${HOME}; git clone -b development https://github.com/MU-Data-Science/EVA.git; ${HOME}/EVA/scripts/setup_bwa_freebayes.sh'
  scp /mydata/hs38.* vm${i}:/mydata/
  echo ${i}
done
