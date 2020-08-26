#!/usr/bin/env bash

cd ${HOME}

# Get bwa
rm -rf bwa
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
rm -rf samtools
git clone https://github.com/samtools/samtools

rm -rf htslib
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
rm -rf freebayes
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

echo "👉 Successful installation of the required tools. 😎"