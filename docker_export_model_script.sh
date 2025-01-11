#!/bin/bash


git clone https://github.com/apsonawane/turnkeyml-cuda.git
cd turnkeyml-cuda
git reset --hard f15fac9638fc03a3a83fff1d14d00e2246edeb63
conda create -n tk-llm python=3.11
source /opt/conda/etc/profile.d/conda.sh
echo "Activating conda environment"
conda activate tk-llm
echo "Installing turnkeyllm"
pip install -e .[llm-oga-cuda]
echo "Installed turnkeyllm"

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

apt-get update

apt-get -y install cudnn9-cuda-12


pip install -y huggingface_hub

echo "Diagnosis:"

conda env list


nvcc --version
nvidia-smi

echo "find libcudnn"
find / -name libcudnn.so.*

echo "Download baseline model"
huggingface-cli download $MODEL_NAME --local-dir /build/oga_models/hf_version/

echo "Running lemonade command"
lemonade -i $MODEL_NAME --cache-dir "/build" oga-load --device cuda --dtype int4 accuracy-mmlu --tests management oga-bench

# echo "Copying the model to ort_src"
ls -la "/build/oga_models/"

echo "lemonade exported onnx model successfully!"

# mkdir /model

# cp -r "/root/oga_models/microsoft_phi-3-mini-4k-instruct/" "/model"

# ls "/model"

# conda create -n rai python==3.12
# conda activate rai

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
# dpkg -i cuda-keyring_1.1-1_all.deb

# apt-get update
# apt-get -y install cudnn9-cuda-12


# echo "Step.1 - installing fastapi uvicorn onnxruntime-genai-cuda"
# pip install fastapi uvicorn onnxruntime-genai-cuda

# echo "step.1b - installing baselina model requirement"
# pip install torch==2.3.1
# pip install flash_attn accelerate==0.31.0 transformers==4.43.0

# echo "Step.2 - azure-cli"
# pip install --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge

# echo "Step.3 - azure-ai"
# pip install azure-ai-evaluation --upgrade

# echo "Step.4 - promptflow-azure"
# pip install promptflow-azure --upgrade

# az login --identity
# echo "print out az account"
# az account show

# cd /ort_src
# python ./docker_main.py
