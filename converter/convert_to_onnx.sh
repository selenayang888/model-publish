#! /bin/bash

# Import base script
source ./turnkeyml.sh

# TODO: rui-ren adapt for cpu version

echo "Installed turnkeyllm"


# Install huggingface_hub
pip install huggingface_hub


if [[ $DEVICE_TYPE == "cpu" ]]; then
    echo "Running on CPU"
    pip install -e .[llm-oga-cpu]

else
    echo "Running on GPU"
    pip install -e .[llm-oga-gpu]

fi

echo "Download baseline model"
# huggingface-cli download $MODEL_NAME --local-dir /build/oga_models/hf_version/

# TODO: rui-ren  we need to add a condition check here
# We need to handle 100 models
huggingface-cli download $MODEL_NAME --local-dir /build/oga_models/hf_version/

echo "Running lemonade command"
# lemonade -i $MODEL_NAME --cache-dir "/build" oga-load --input_path "/build/phi-4-mini-instruct-01072025/hf_version" --device cuda --dtype int4 accuracy-mmlu --tests management

echo "lemonade -i $MODEL_NAME --cache-dir "/build" oga-load --device $DEVICE_TYPE --dtype $DATA_TYPE"

lemonade -i $MODEL_NAME --cache-dir "/build" oga-load --device $DEVICE_TYPE --dtype $DATA_TYPE

# echo "Copying the model to ort_src"

ls -la "/build/oga_models/"

# remove the hf_version and do not upload to the azure blob
rm -rf /build/oga_models/hf_version

echo "lemonade exported onnx model successfully!"