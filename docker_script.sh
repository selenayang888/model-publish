#!/bin/bash


pip install fastapi
pip install uvicorn
pip install onnxruntime-genai-cuda
pip install pandas
pip install --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge
pip install azure-ai-evaluation --upgrade
pip install promptflow-azure --upgrade


nvcc --version
nvidia-smi

cd /ort_src
ls ./
python ./docker_main.py
