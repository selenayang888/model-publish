#!/bin/bash

pip install --upgrade pip

# Install Turnkeyml
git clone https://github.com/rui-ren/turnkeyml.git
cd turnkeyml
conda create -n tk-llm python=3.10
source /opt/conda/etc/profile.d/conda.sh
echo "Activating conda environment"
conda activate tk-llm
echo "Installing turnkeyllm"


if [[ $DEVICE_TYPE == "cpu" ]]; then
    echo "Install packages for onnx-cpu"
    pip install -r docker/requirements-onnx-cpu.txt

elif [[ $DEVICE_TYPE == "cuda" ]]; then
    echo "Install packages for onnx-cuda"
    pip install -r docker/requirements-onnx-cuda.txt

else
    echo "Install packages for torch"
    pip install -r docker/requirements-torch.txt

fi