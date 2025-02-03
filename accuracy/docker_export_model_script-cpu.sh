#!/bin/bash

# Import Trunkeyml Base scripts
./turnkeyml.sh

# Install Turnkeyml
echo "Installed turnkeyllm"
pip install -e .[llm-oga-cpu]

pip list

# Run MMLU accuracy test
echo "Running lemonade command"
lemonade -i $MODEL_NAME --cache-dir "/build" oga-load --device cuda --dtype int4 accuracy-mmlu

echo "lemonade run accuracy successfully!"

