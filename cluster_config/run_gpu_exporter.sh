#!/bin/bash

# Remove older installations
sudo nv-hostengine -t
sudo apt remove datacenter-gpu-manager -y

# Remove old GPG key
# sudo apt-key del 7fa2af80 -y
# This seems to be causing problems on cloudlab

# Determine distribution name
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')

# Get the repository
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install DCGM
sudo apt-get install -y datacenter-gpu-manager

# Enable the DCGM service and start it now
sudo systemctl --now enable nvidia-dcgm

# Verify installation
dcgmi discovery -l
sudo apt install docker.io
sudo docker run -d --gpus all --rm -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:3.2.5-3.1.8-ubuntu20.04

# Make sure to kill the docker container when you are done 