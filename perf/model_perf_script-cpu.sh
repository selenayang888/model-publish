
#!/bin/bash

# Import Trunkeyml Base scripts
./turnkeyml.sh

# Install Turnkeyml
echo "Installed turnkeyllm"

if [[ $DEVICE_TYPE == "cpu" ]]; then

    pip install -e .[llm-oga-cpu]

else 
    pip install -e .[llm-oga-gpu]

fi

# Run MMLU accuracy test
echo "Running lemonade command"
lemonade -i $MODEL_NAME --cache-dir "/build" oga-load --device ${DEVICE_TYPE} --dtype ${DATA_TYPE} accuracy-mmlu

echo "lemonade run accuracy successfully!"

