#!/bin/bash

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

apt-get update
apt-get install -y
apt-get -y install cudnn9-cuda-12

pip install -y huggingface_hub


nvcc --version
nvidia-smi

echo "find libcudnn"
find / -name libcudnn.so.*

git clone https://github.com/apsonawane/turnkeyml-cuda.git
cd turnkeyml-cuda
git reset --hard f15fac9638fc03a3a83fff1d14d00e2246edeb63
conda create -n tk-llm python=3.10
source /opt/conda/etc/profile.d/conda.sh
echo "Activating conda environment"
conda activate tk-llm
conda env list
echo "Installing turnkeyllm"
pip install -e .[llm-oga-cuda]
echo "Installed turnkeyllm"

echo "Download baseline model"
huggingface-cli download $MODEL_NAME --local-dir /build/oga_models/hf_version/

echo "Running lemonade command"
lemonade -i $MODEL_NAME --cache-dir "/build" oga-load --device cuda --dtype int4 accuracy-mmlu --tests management oga-bench

# echo "Copying the model to ort_src"
ls -la "/build/oga_models/"

echo "lemonade exported onnx model successfully!"