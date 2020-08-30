#!/usr/bin/env bash

DATA_DIR="$1"
SHARE_DIR="$2"

# get bwa
tar -xvf $SHARE_DIR/EVA_Tools/bwa.tar -C $DATA_DIR
ln -sf $DATA_DIR/bwa $HOME/bwa
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
sudo echo 'PATH:=$(HOMEBREW_PREFIX)/bin:$(PATH)' >> $HOME/abyss-pe
mkdir -p $DATA_DIR/tmp

echo "👉 Successful installation of the required tools. 😎"
