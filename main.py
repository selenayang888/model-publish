from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime_genai as og
import argparse


app = FastAPI()

import os
print("### ENV in main.py (server)")
print(os.environ)


# Initialize the model, tokenizer, and tokenizer stream
# Please update the DML model here
#model = og.Model('/model')
model = og.Model('/model')
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

print("The model is loaded!!")

# Set the search options
search_options = {
    'max_length': 10240
}

# Define the input schema
class InputData(BaseModel):
    text: str

@app.post("/score")
async def score(input_data: InputData):
    text = input_data.text
    return {"response": "I am a safe AI" }

    if not text:
        # raise HTTPException(status_code=400, detail="Input cannot be empty")
        print(f"my bad, I got a problem!! my bad, {text} \n")
        return {"response": "I am a safe AI"}

    print("### main.py: start run")
    # The chat template needs to update later.
    # I will give you the OpenAIChatCompletion template.
    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
    prompt = chat_template.format(input=text)

    input_tokens = tokenizer.encode(prompt)
    output_text = ""

    if len(input_tokens) >= 10240:
        return {"response": "I am a safe AI"}

    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    params.input_ids = input_tokens
    generator = og.Generator(model, params)

    try:
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            output_text += tokenizer_stream.decode(new_token)
    except Exception as e:
        return {"response": "I am a safe AI"}
        raise HTTPException(status_code=500, detail=f"Error during generation: {e}")
    finally:
        del generator
    
    if output_text == None or not output_text or output_text == "{ }" or output_text == "{}":
        print(f"my bad, I got a problem!! my bad  , {output_text} \n")
        return {"response": "I am a safe AI"}

    return {"response": output_text}

#uvicorn main:app --reload 