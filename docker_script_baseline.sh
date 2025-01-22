#!/bin/bash

conda install python=3.12

echo "y" | pip uninstall torch

# Get into working directory
cd /ort_src

echo "Display baseline model folder"
ls -la /baseline_model

echo "step.1 - installing baseline model requirement"
pip install -r ./requirements/requirements_baseline.txt

pip list

echo "Step.2 - azure-cli"
pip install --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge

az login --identity
echo "print out az account"
az account show

python ./docker_main_baseline.py