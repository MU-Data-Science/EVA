#!/usr/bin/env bash

DATA_DIR="$1"

# htop installation.
sudo apt-get install htop

# Get bwa
git clone https://github.com/lh3/bwa $DATA_DIR/bwa
make -C ${DATA_DIR}/bwa
echo "👉 Done with bwa setup 😎"

# get brew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)" < /dev/null
echo 'eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)' >> /users/${USER}/.profile
eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)
echo "👉 Done with Brew installation 😎"

# get sambamba
brew install brewsci/bio/sambamba
ln -sf /home/linuxbrew/.linuxbrew/bin/sambamba .
echo "👉 Done with Sambamba installation 😎"

# get samtools and hstlib
git clone https://github.com/samtools/samtools ${DATA_DIR}/samtools
git clone https://github.com/samtools/htslib ${DATA_DIR}/htslib

cd ${DATA_DIR}/htslib
sudo apt-get update
sudo apt-get install libbz2-dev -y
sudo apt-get install liblzma-dev -y
autoheader; autoconf -Wno-syntax; ./configure; make
sudo make install
echo "👉 Done with Hstlib installation 😎"

cd ${DATA_DIR}/samtools
autoheader; autoconf -Wno-syntax; ./configure; make
sudo make install

echo "👉 Done with Samtools installation 😎"

# get freebayes
git clone --recursive git://github.com/ekg/freebayes.git ${DATA_DIR}/freebayes
sudo apt-get install cmake -y
cd ${DATA_DIR}/freebayes
make
cd
echo "👉 Done with Freebayes installation 😎"

# setup Picard
PICARD_VERSION=2.23.2
PICARD_JAR_RELEASE=https://github.com/broadinstitute/picard/releases/download/${PICARD_VERSION}/picard.jar
wget ${PICARD_JAR_RELEASE} -O ${DATA_DIR}/
echo "👉 Done with Picard download 😎"

# setup GATK
GATK_VERSION=4.1.8.0
GATK_LOCAL_ZIP=gatk.zip
GATK_ZIP_RELEASE=https://github.com/broadinstitute/gatk/releases/download/${GATK_VERSION}/gatk-${GATK_VERSION}.zip
wget ${GATK_ZIP_RELEASE} -O ${DATA_DIR}/${GATK_LOCAL_ZIP}
unzip ${DATA_DIR}/${GATK_LOCAL_ZIP} -d ${DATA_DIR}/gatk
rm -rf ${DATA_DIR}/${GATK_LOCAL_ZIP}
echo "👉 Done with GATK download 😎"

echo "👉 Successful installation of the required tools. 😎"