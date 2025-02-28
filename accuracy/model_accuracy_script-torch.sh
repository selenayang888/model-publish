# TODO: rui-ren

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

# Improt Turnkeyml Base scripts
./turnkeyml.sh
pip install -e .[llm-oga-cuda]
echo "Installed turnkeyllm"

pip list

# Run MMLU accuracy test
echo "Running lemonade command"
lemonade -i $MODEL_NAME --cache-dir "/build" oga-load --device ${DEVICE_TYPE} --dtype ${DATA_TYPE} accuracy-mmlu

echo "lemonade test CUDA accuracy mmlu successfully!"

