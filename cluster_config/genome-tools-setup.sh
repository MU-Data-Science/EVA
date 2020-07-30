#!/usr/bin/env bash

DATA_DIR="$1"
SHARE_DIR="$2"

# htop installation.
sudo apt-get install htop

# get bwa
tar -xvf $SHARE_DIR/EVA_Tools/bwa.tar -C $DATA_DIR
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

echo "👉 Done with Samtools and Htslib installation 😎"

# get freebayes
tar -xvf $SHARE_DIR/EVA_Tools/freebayes.tar -C $DATA_DIR
echo "👉 Done with Freebayes installation 😎"

# setup Picard
cp $SHARE_DIR/EVA_Tools/picard.jar $DATA_DIR
echo "👉 Done with Picard copy 😎"

# setup GATK
GATK_VERSION=4.1.8.0
unzip $SHARE_DIR/EVA_Tools/gatk-${GATK_VERSION}.zip -d $DATA_DIR
h
echo "👉 Successful installation of the required tools. 😎"