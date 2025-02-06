#!/bin/bash

conda install python=3.12

apt-get update

echo "Step 1 - Install dependencies"
pip install -r requirements-onnx-cpu.txt


echo "Step 2 - Login Azure"
az login --identity
echo "print out az account"
az account show

