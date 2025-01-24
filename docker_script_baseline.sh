#!/bin/bash

conda install python=3.12

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

apt-get update
apt-get -y install cudnn9-cuda-12


echo "Step.1 - installing fastapi uvicorn onnxruntime-genai==0.6.0rc1"
pip install fastapi uvicorn onnxruntime-genai==0.6.0rc1

echo "step.1b - installing baselina model requirement"
pip install torch==2.3.1
pip install flash_attn accelerate==0.31.0 transformers

echo "Step.2 - azure-cli"
pip install --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge

echo "Step.3 - azure-ai and promptflow"
pip install azure-ai-evaluation --upgrade promptflow-azure marshmallow==3.23.2


az login --identity
echo "print out az account"
az account show

cd /ort_src
python ./docker_main_baseline.py