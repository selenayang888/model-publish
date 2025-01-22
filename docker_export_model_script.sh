#!/bin/bash

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
# dpkg -i cuda-keyring_1.1-1_all.deb

# apt-get update
# apt-get install -y
# apt-get -y install cudnn9-cuda-12


pip install huggingface_hub

echo "Diagnosis:"

conda env list


# nvcc --version
nvidia-smi

# echo "find libcudnn"
# find / -name libcudnn.so.*

# git clone https://github.com/rui-ren/turnkeyml.git
# cd turnkeyml
# conda create -n tk-llm python=3.10
# source /opt/conda/etc/profile.d/conda.sh
# echo "Activating conda environment"
# conda activate tk-llm
# echo "Installing turnkeyllm"
# pip install -e .[llm-oga-cuda]
# echo "Installed turnkeyllm"

echo "Download baseline model"
# huggingface-cli download $MODEL_NAME --local-dir /build/oga_models/hf_version/

# TODO: rui-ren  we need to add a condition check here 
huggingface-cli download $MODEL_NAME --local-dir /mnt/oga_models/hf_version/

echo "Running lemonade command"
# lemonade -i $MODEL_NAME --cache-dir "/build" oga-load --input_path "/build/phi-4-mini-instruct-01072025/hf_version" --device cuda --dtype int4 accuracy-mmlu --tests management

# TODO: rui-ren add conditions to disable the benchmarking
# lemonade -i $MODEL_NAME --cache-dir "/build" oga-load --device cuda --dtype int4 accuracy-mmlu --tests management


# echo "Copying the model to ort_src"

ls -la "/mnt/oga_models/hf_version"

# Here need to update
## cp -r /build/phi-4-mini-instruct-01072025/hf_version/* /build/oga_models/baseline_model/

ls -la "/mnt/oga_models/"

echo "lemonade exported onnx model successfully!"
