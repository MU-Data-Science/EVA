#!/usr/bin/env bash

# Configurations.
CONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh"
DATA_DIR=${1}

# Download & Install Conda.
wget ${CONDA_URL} -O ${DATA_DIR}/Anaconda3.sh
bash ${DATA_DIR}/Anaconda3.sh -b -p ${DATA_DIR}/Anaconda3

# Initialize Conda.
${DATA_DIR}/Anaconda3/bin/conda init