from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime_genai as og
import argparse


app = FastAPI()

import os

print("### ENV in main.py (server)")
print(os.environ)


model = og.Model("./phi-4-omni/cuda/cuda-int4-rtn-block-32")

processor = model.create_multimodal_processor()
tokenizer_stream = processor.create_stream()

print("The model is loaded!!")

# Set the search options
search_options = {"max_length": 2048}


# Define the input schema
class InputData(BaseModel):
    # text: str
    audios_paths: str


@app.post("/score")
async def score(input_data: InputData):
    # text = input_data.text
    text = None
    audios_paths = input_data.audios_paths

    print("### main.py: start run")
    # The chat template needs to update later.
    # I will give you the OpenAIChatCompletion template.

    prompt = "<|user|>\n"

    # Get audios
    if len(audios_paths) == 0:
        print("No audio provided")
    else:
        if not os.path.exists(audios_paths):
            raise FileNotFoundError(f"Audio file not found: {audios_paths}")
            print(f"Using audio: {audios_paths}")
        
        prompt += f"<|audio_1|>\n"
        audios = og.Audios.open(audios_paths)

    # Get Text
    text = None

    prompt += f"{text}<|end|>\n<|assistant|>\n"

    audios = og.Audios.open(audios_paths)

    inputs = processor(prompt, images=None, audios=audios)
    output_text = ""

    params = og.GeneratorParams(model)
    params.set_inputs(inputs)

    params.set_search_options(**search_options)
    generator = og.Generator(model, params)

    try:
        while not generator.is_done():

            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]

            output_text += tokenizer_stream.decode(new_token)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during generation: {e}")
    finally:
        del generator

    return {"response": output_text}
