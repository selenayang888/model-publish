#!/bin/bash


pip install fastapi
pip install uvicorn
pip install onnxruntime-genai-cuda

cd /ort_src
python ./docker_main.py
