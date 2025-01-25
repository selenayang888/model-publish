#!/bin/bash

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

apt-get update
apt-get install -y
apt-get -y install cudnn9-cuda-12


# echo "Step.1 - onnxruntime-genai-cuda"
# pip install onnxruntime-genai-cuda


pip install huggingface_hub

echo "Diagnosis:"

conda env list


nvcc --version
nvidia-smi

echo "find libcudnn"
find / -name libcudnn.so.*

git clone https://github.com/rui-ren/turnkeyml.git
cd turnkeyml
conda create -n tk-llm python=3.10
source /opt/conda/etc/profile.d/conda.sh
echo "Activating conda environment"
conda activate tk-llm
echo "Installing turnkeyllm"
pip install -e .[llm-oga-cpu]
echo "Installed turnkeyllm"

pip list

echo "Download baseline model"
# huggingface-cli download $MODEL_NAME --local-dir /build/oga_models/hf_version/


echo "Running lemonade command"
lemonade -i $MODEL_NAME --cache-dir "/build" oga-load --input_path "/build/phi-4-mini-instruct-01222025/hf_version" --device cpu --dtype fp16


ls -la "/build/oga_models/"

mkdir /build/oga_models/baseline_model/

cp -r /build/phi-4-mini-instruct-01222025/hf_version/* /build/oga_models/baseline_model/

ls -la "/build/oga_models/"

echo "lemonade exported onnx model successfully!"

