#!/bin/bash

conda install python=3.12

apt-get update

echo "Step 1 - Install dependencies"
pip install -r requirements-onnx-cpu.txt


echo "Step 2 - Login Azure"
az login --identity
echo "print out az account"
az account show


# 1. Start the backend server

if [[ $DEVICE_TYPE == "cpu" ]]; then

    python backend/server.py --device cpu

elif [[ $DEVICE_TYPE == "gpu" ]]; then
    python backend/server.py --device gpu

else
    echo "Invalid device type"
    python backend/server_pytorch.py
fi


# 2. Run the RAI process

if [[ "eci" in $RAI_TESTS]]; then
    echo "Running ECI tests"
    python run_eci.py
fi

if [[ "ip" in $RAI_TESTS]]; then
    echo "Running IP tests"
    python run_ip.py
fi

if [[ "groundness" in $RAI_TESTS]]; then
    echo "Running Groundness tests"
    python run_groundness.py
fi

if [[ "jailbreak" in $RAI_TESTS]]; then
    echo "Running Jailbreak tests"
    python run_jailbreak.py
fi

if [[ "safety" in $RAI_TESTS]]; then
    echo "Running Safety tests"
    python run_safety.py
fi

if [[ "summarization" in $RAI_TESTS]]; then
    echo "Running Summarization tests"
    python run_summarization.py
fi