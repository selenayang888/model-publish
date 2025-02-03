#!/bin/bash

echo "Diagnosis:"
conda env list

# Install Turnkeyml
git clone https://github.com/rui-ren/turnkeyml.git
cd turnkeyml
conda create -n tk-llm python=3.10
source /opt/conda/etc/profile.d/conda.sh
echo "Activating conda environment"
conda activate tk-llm
echo "Installing turnkeyllm"