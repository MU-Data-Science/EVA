#!/usr/bin/env bash

cd ${HOME}

# Get bwa
git clone https://github.com/lh3/bwa
cd bwa
make
cd ${HOME}
echo "👉 Done with bwa setup 😎"

# get freebayes
cd  ${HOME}
git clone --recursive git://github.com/ekg/freebayes.git
sudo apt-get install cmake -y
cd freebayes
make
cd  ${HOME}
echo "👉 Done with Freebayes installation 😎"

# setup Java
sudo apt-get update
sudo apt-get install default-jre -y
sudo apt-get install default-jdk -y
echo "👉 Done with Java installation 😎"

echo "👉 Successful installation of the required tools. 😎"