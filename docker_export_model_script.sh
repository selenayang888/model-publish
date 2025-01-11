#!/bin/bash


# git clone https://github.com/apsonawane/turnkeyml-cuda.git
# cd turnkeyml-cuda
# git reset --hard f15fac9638fc03a3a83fff1d14d00e2246edeb63
conda create -n tk-llm python=3.10
# source /opt/conda/etc/profile.d/conda.sh
echo "Activating conda environment"
conda activate tk-llm
echo "Installing turnkeyllm"
# pip install -e .[llm-oga-cuda]
echo "Installed turnkeyllm"

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

apt-get update
apt-get -y install cudnn9-cuda-12


# echo "Step.1 - onnxruntime-genai-cuda"
# pip install onnxruntime-genai-cuda


pip install -y huggingface_hub

echo "Diagnosis:"

conda env list


nvcc --version
nvidia-smi

echo "find libcudnn"
find / -name libcudnn.so.*
