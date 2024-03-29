#!/usr/bin/env bash

DATA_DIR="$1"
SHARE_DIR="$2"

# get bwa
tar -xvf $SHARE_DIR/EVA_Tools/bwa.tar -C $DATA_DIR
ln -sf $DATA_DIR/bwa $HOME/bwa
echo "👉 Done with bwa setup 😎"

# Setup bwa-mem2
#git clone --recursive https://github.com/bwa-mem2/bwa-mem2
#cd bwa-mem2
#make
tar -xvf $SHARE_DIR/EVA_Tools/bwa-mem2.tar -C $DATA_DIR
ln -sf $DATA_DIR/bwa-mem2 $HOME/bwa-mem2
echo "👉 Done with bwa-mem2 setup 😎"

# get brew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)" </dev/null
echo 'eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)' >>/users/${USER}/.profile
eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)
echo "👉 Done with Brew installation 😎"

# get sambamba
brew install brewsci/bio/sambamba
ln -sf /home/linuxbrew/.linuxbrew/bin/sambamba .
echo "👉 Done with Sambamba installation 😎"

# get samtools and htslib
tar -xvf $SHARE_DIR/EVA_Tools/samtools.tar -C $DATA_DIR
tar -xvf $SHARE_DIR/EVA_Tools/htslib.tar -C $DATA_DIR
ln -sf $DATA_DIR/samtools $HOME/samtools
echo "👉 Done with Samtools and Htslib installation 😎"

# get freebayes
tar -xvf $SHARE_DIR/EVA_Tools/freebayes.tar -C $DATA_DIR
ln -sf $DATA_DIR/freebayes $HOME/freebayes
echo "👉 Done with Freebayes installation 😎"

# setup Picard
cp $SHARE_DIR/EVA_Tools/picard.jar $DATA_DIR
ln -sf $DATA_DIR/picard.jar $HOME/picard.jar
echo "👉 Done with Picard copy 😎"

# setup GATK
GATK_VERSION=4.1.8.0
unzip $SHARE_DIR/EVA_Tools/gatk-${GATK_VERSION}.zip -d $DATA_DIR
ln -sf $DATA_DIR/gatk-${GATK_VERSION} $HOME/gatk-${GATK_VERSION}
mkdir $DATA_DIR/gatk-tmp
echo "👉 Done with GATK copy 😎"

# setup SPAdes
SPADES_VERSION=3.14.1
if [ ! -f $SHARE_DIR/EVA_Tools/SPAdes-${SPADES_VERSION}-Linux.tar.gz ]; then
  wget http://cab.spbu.ru/files/release${SPADES_VERSION}/SPAdes-${SPADES_VERSION}-Linux.tar.gz -P $SHARE_DIR/EVA_Tools/
fi
tar -xzf $SHARE_DIR/EVA_Tools/SPAdes-${SPADES_VERSION}-Linux.tar.gz -C $DATA_DIR
ln -sf $DATA_DIR/SPAdes-${SPADES_VERSION}-Linux $HOME/spades
echo "👉 Done with SPAdes setup 😎"

# Install Abyss
brew install abyss
ABYSS_PE_PATH=/home/linuxbrew/.linuxbrew/bin
ln -sf ${ABYSS_PE_PATH}/abyss-pe $HOME/abyss-pe
sudo chmod +w ${ABYSS_PE_PATH}/abyss-pe
sudo echo 'PATH:=$(HOMEBREW_PREFIX)/bin:$(PATH)' >>${ABYSS_PE_PATH}/abyss-pe
mkdir -p $DATA_DIR/tmp
echo "👉 Done with Abyss setup 😎"

# Setup Adam.
git clone https://github.com/Arun-George-Zachariah/adam.git $DATA_DIR/adam
echo "export ADAM_HOME=$DATA_DIR/adam" >>~/.bashrc
echo "👉 Done with Adam setup 😎"

# Setup Cannoli.
git clone https://github.com/Arun-George-Zachariah/cannoli.git $DATA_DIR/cannoli
echo "export CANNOLI_HOME=$DATA_DIR/cannoli" >>~/.bashrc
echo "👉 Done with Cannoli setup 😎"

# Installing GATK python dependencies.
$DATA_DIR/Anaconda3/bin/conda env create -n gatk -f ~/gatk-${GATK_VERSION}/gatkcondaenv.yml

# Setup EVA
git clone https://github.com/MU-Data-Science/EVA.git $HOME/EVA
echo "👉 Done cloning EVA 😎"

echo "👉 Successful installation of the required tools. 😎"
