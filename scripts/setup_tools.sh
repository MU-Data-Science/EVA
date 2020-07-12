#!/usr/bin/env bash

cd ${HOME}

# Get bwa
git clone https://github.com/lh3/bwa
cd bwa
make
cd ${HOME}
echo "👉 Done with bwa setup 😎"

# get brew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)" < /dev/null
echo 'eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)' >> /users/${USER}/.profile
eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)

echo "👉 Done with Brew installation 😎"

# get sambamba
brew install brewsci/bio/sambamba
cd ${HOME}
ln -sf /home/linuxbrew/.linuxbrew/bin/sambamba .
echo "👉 Done with Sambamba installation 😎"

# get samtools and hstlib
git clone https://github.com/samtools/samtools
git clone https://github.com/samtools/htslib

cd htslib
sudo apt-get update
sudo apt-get install libbz2-dev -y
sudo apt-get install liblzma-dev -y
autoheader; autoconf -Wno-syntax; ./configure; make
sudo make install
echo "👉 Done with Hstlib installation 😎"

cd ../samtools
autoheader; autoconf -Wno-syntax; ./configure; make
sudo make install

echo "👉 Done with Samtools installation 😎"

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

# setup Picard
cd ${HOME}
PICARD_JAR_RELEASE=https://github.com/broadinstitute/picard/releases/download/2.23.2/picard.jar
wget ${PICARD_JAR_RELEASE}
echo "👉 Done with Picard download 😎"

# setup GATK
cd ${HOME}
GATK_ZIP_RELEASE=https://github.com/broadinstitute/gatk/releases/download/4.1.8.0/gatk-4.1.8.0.zip
wget ${GATK_ZIP_RELEASE}
unzip ${GATK_ZIP_RELEASE}
echo "👉 Done with GATK download 😎"

echo "👉 Successful installation of the required tools. 😎"