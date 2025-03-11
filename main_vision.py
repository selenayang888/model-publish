from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, RootModel
import onnxruntime_genai as og
import os
import argparse
import base64
from urllib.parse import urlsplit

def decode_data_url(data_url):
    # Split the URL into components
    scheme, netloc, path, query, fragment = urlsplit(data_url)
    # Ensure it's a data URL
    if scheme != 'data':
        raise ValueError("Not a data URL")
    # Split the data URL into metadata and data
    metadata, encoded_data = path.split(',', 1)
    # Decode the base64 data
    if 'base64' in metadata:
        decoded_data = base64.b64decode(encoded_data)
    else:
        decoded_data = encoded_data
    return decoded_data


app = FastAPI()

# Initialize the model, tokenizer, and tokenizer stream
# Please update the DML model here
#model = og.Model('/workspace/rai/phi-3.5-vision-cuda-int4-rtn-block-32')
model = og.Model("./phi-4-omni/cuda/cuda-int4-rtn-block-32")

#tokenizer = og.Tokenizer(model)
#tokenizer_stream = tokenizer.create_stream()
processor = model.create_multimodal_processor()
tokenizer_stream = processor.create_stream()

print("The model is loaded!!")

# Set the search options
search_options = {
    'max_length': 2048
}

# # Define the input schema
# class InputData(BaseModel):
#     text: str

# class VisionModelInputData(RootModel):
#     root: list[dict]
class VisionModelInputData(BaseModel):
    request: list[dict]

@app.post("/score")
async def score(input_data: VisionModelInputData):
    # TODO: now I got this input_data and I need to pre-process it to generate correct data to feed
    #       to OrtGenai
    #print(input_data)
    image_paths = []
    text = ""
    for input in input_data.request:
        if input["type"] == "image_url":
            image_paths.append(input["image_url"]["url"])
        else:
            text += input["text"]
    
    image_paths = [decode_data_url(image_path) for image_path in image_paths]

    images = None
    prompt = "<|user|>\n"
    if len(image_paths) == 0:
        print("No image provided")
    else:
        for i, image_path in enumerate(image_paths):
            
            
            prompt += f"<|image_{i+1}|>\n"

        images = og.Images.open_bytes(*image_paths)
        

    prompt += f"{text}<|end|>\n<|assistant|>\n"
    print("Processing images and prompt...")
    inputs = processor(prompt, images=images, audios=None)

    print("Generating response...")
    params = og.GeneratorParams(model)
    params.set_inputs(inputs)
    params.set_search_options(max_length=12548)

    generator = og.Generator(model, params)
    
    #params = og.GeneratorParams(model)
    #params.set_search_options(**search_options)
    # params.input_ids = input_tokens
    #generator = og.Generator(model, params)

    output_text = ""
    if len(inputs) > 12548:
        return {"response": "I am a safe AI Robot"}
    
    try:
        while not generator.is_done():
            #generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            output_text += tokenizer_stream.decode(new_token)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during generation: {e}")
    finally:
        del generator

    return {"response": output_text}

#  uvicorn main_vision:app --reload