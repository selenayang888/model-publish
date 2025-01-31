from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse


app = FastAPI()

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "/baseline_model",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("/baseline_model")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

print("The baseline model is loaded!!")

# Set the search options
search_options = {"max_length": 4096}


# Define the input schema
class InputData(BaseModel):
    text: str


@app.get("/health")
async def healthcheck():
    return {"status": 200}


@app.post("/score")
async def score(input_data: InputData):
    text = input_data.text

    print("### main_baseline.py: start run")
    # The chat template needs to update later.
    # I will give you the OpenAIChatCompletion template.
    chat_template = "<|user|>\n{input} <|end|>\n<|assistant|>"
    prompt = chat_template.format(input=text)

    messages = [
        {"role": "user", "content": text},
    ]

    generation_args = {
        "max_new_tokens": 2048,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    output_text = pipe(messages, **generation_args)[0]["generated_text"]

    return {"response": output_text}


# uvicorn main:app --reload
