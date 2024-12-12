#!/bin/bash


pip install fastapi
pip install uvicorn
pip install onnxruntime-genai-cuda
pip install pandas

cd /ort_src
python ./docker_main.py
