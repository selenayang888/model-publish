#!/bin/bash
conda activate ptca

echo "Step.1 - installing fastapi uvicorn onnxruntime-genai-cuda pandas"
pip install fastapi uvicorn onnxruntime-genai-cuda pandas

echo "Step.2 - azure-cli"
pip install --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge

echo "Step.3 - azure-ai"
pip install azure-ai-evaluation --upgrade

echo "Step.4 - promptflow-azure"
pip install promptflow-azure --upgrade

echo "Diagosis:"

conda env list

echo "pip list:"
pip list

nvcc --version
nvidia-smi

cd /ort_src

python ./docker_main.py
