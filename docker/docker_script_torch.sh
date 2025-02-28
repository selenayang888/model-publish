#!/bin/bash

conda install python=3.12

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

apt-get update
apt-get -y install cudnn9-cuda-12


echo "Step 1 - installing dependencies"
pip install -r reuirements-pytorch.txt


az login --identity
echo "print out az account"
az account show
