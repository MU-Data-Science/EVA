#!/usr/bin/env bash

CONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh"

# Download & Install Conda.
wget $CONDA_URL -O $DATA_DIR/Anaconda3.sh
bash $DATA_DIR/Anaconda3.sh -b -p $DATA_DIR/anaconda3

# Initialize Conda.
conda init

# Installing GATK dependencies.
conda env create -n gatk -f gatkcondaenv.yml