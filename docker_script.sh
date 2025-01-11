#!/bin/bash

conda install python=3.12

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

apt-get update
apt-get -y install cudnn9-cuda-12

echo "check CUDA version"
nvcc --version
nvidia-smi

