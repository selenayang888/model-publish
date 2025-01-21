#!/bin/bash

conda install python=3.12



echo "step.1 - installing baseline model requirement"
pip install -r ./requirements/requirements_baseline.txt

echo "Step.2 - azure-cli"
pip install --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge

az login --identity
echo "print out az account"
az account show

cd /ort_src
python ./docker_main_baseline.py