#!/bin/bash

conda install python=3.12


echo "Step.1 - installing fastapi uvicorn onnxruntime-genai-cuda"
pip install fastapi uvicorn onnxruntime-genai-cuda

echo "step.1b - installing baselina model requirement"
pip install torch==2.5.1

# ACPT will cover it.

pip install flash_attn==2.7.3 accelerate==0.31.0 transformers==4.48.1

echo "Step.2 - azure-cli"
pip install --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge

echo "Step.3 - azure-ai and promptflow"
pip install azure-ai-evaluation --upgrade promptflow-azure marshmallow==3.23.2


az login --identity
echo "print out az account"
az account show

cd /ort_src
python ./docker_main_baseline.py