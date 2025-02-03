#! /bin/bash

# Import base script
source ../turnkeyml.sh

# TODO: rui-ren adapt for cpu version
pip install -e .[llm-oga-gpu]
echo "Installed turnkeyllm"

echo "Download baseline model"
# huggingface-cli download $MODEL_NAME --local-dir /build/oga_models/hf_version/

# TODO: rui-ren  we need to add a condition check here 
huggingface-cli download $MODEL_NAME --local-dir /build/oga_models/hf_version/

echo "Running lemonade command"
# lemonade -i $MODEL_NAME --cache-dir "/build" oga-load --input_path "/build/phi-4-mini-instruct-01072025/hf_version" --device cuda --dtype int4 accuracy-mmlu --tests management

lemonade -i $MODEL_NAME --cache-dir "/build" oga-load --device cpu --dtype int4


# echo "Copying the model to ort_src"

ls -la "/build/oga_models/"

mkdir /build/oga_models/baseline_model/


ls -la "/build/oga_models/"

echo "lemonade exported onnx model successfully!"