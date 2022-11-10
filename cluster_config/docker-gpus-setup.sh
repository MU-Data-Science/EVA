#!/usr/bin/env bash

DATA_DIR="$1"
UBUNTU_VERSION="ubuntu1804"
CUDA_VERSION="11.0.2"
CUDA_PACKAGE="cuda-11.0"
DRIVER_NAME="450.51.05-1_amd64"
PARABRICKS="nvcr.io/nvidia/clara/clara-parabricks:4.0.0-1"

cd ${DATA_DIR}

echo "👉Installing CUDA ${CUDA_VERSION} for ${UBUNTU_VERSION}..."
wget https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/x86_64/cuda-${UBUNTU_VERSION}.pin
sudo mv cuda-${UBUNTU_VERSION}.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/cuda-repo-${UBUNTU_VERSION}-11-0-local_${CUDA_VERSION}-${DRIVER_NAME}.deb
sudo dpkg -i cuda-repo-${UBUNTU_VERSION}-11-0-local_${CUDA_VERSION}-${DRIVER_NAME}.deb
sudo apt-key add /var/cuda-repo-${UBUNTU_VERSION}-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install ${CUDA_PACKAGE}
echo "👉 Done! A reboot is needed to load the driver."

echo "👉 Installing Docker..."
sudo apt update
sudo apt install docker.io -y
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl start docker
sudo sed -i -e 's/dockerd -H/dockerd -g \/mydata\/docker -H/g' /lib/systemd/system/docker.service
sudo systemctl daemon-reload
sudo systemctl start docker
sudo docker pull ${PARABRICKS}
echo "👉 Done!"
