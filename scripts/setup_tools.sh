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
git submodule update --init --recursive
autoheader; autoconf -Wno-syntax; ./configure; make
sudo make install
echo "👉 Done with Hstlib installation 😎"

cd ../samtools
sudo apt-get install libncurses5-dev libncursesw5-dev -y
autoheader; autoconf -Wno-syntax; ./configure; make
sudo make install

echo "👉 Done with Samtools installation 😎"

# get freebayes
cd  ${HOME}
git clone --recursive https://github.com/freebayes/freebayes.git
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
PICARD_VERSION=2.23.2
PICARD_JAR_RELEASE=https://github.com/broadinstitute/picard/releases/download/${PICARD_VERSION}/picard.jar
wget ${PICARD_JAR_RELEASE}
echo "👉 Done with Picard download 😎"

# setup GATK
cd ${HOME}
GATK_VERSION=4.1.8.0
GATK_LOCAL_ZIP=gatk.zip
GATK_ZIP_RELEASE=https://github.com/broadinstitute/gatk/releases/download/${GATK_VERSION}/gatk-${GATK_VERSION}.zip
wget ${GATK_ZIP_RELEASE} -O ${GATK_LOCAL_ZIP}
unzip ${GATK_LOCAL_ZIP}
rm -rf ${GATK_LOCAL_ZIP}
echo "👉 Done with GATK download 😎"

# setup SPAdes
cd ${HOME}
SPADES_VERSION=3.14.1
wget http://cab.spbu.ru/files/release${SPADES_VERSION}/SPAdes-${SPADES_VERSION}-Linux.tar.gz
tar -xzf SPAdes-${SPADES_VERSION}-Linux.tar.gz
mv SPAdes-${SPADES_VERSION}-Linux spades
rm -rf SPAdes-${SPADES_VERSION}-Linux.tar.gz
echo "👉 Done with SPAdes download 😎"

# Install Abyss
brew install abyss

# START setup
cd ${HOME} && git clone https://github.com/alexdobin/STAR.git

echo "👉 Successful installation of the required tools. 😎"