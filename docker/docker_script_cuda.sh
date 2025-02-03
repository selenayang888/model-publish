#!/bin/bash

conda install python=3.12

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

apt-get update
apt-get -y install cudnn9-cuda-12


echo "Step 1 Install dependencies"
pip install -r requirements-onnx-cuda.txt


echo "Step 2 Login Azure"
az login --identity
echo "print out az account"
az account show

cd /ort_src
python ./docker_main.py