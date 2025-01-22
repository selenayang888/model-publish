from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse
from pathlib import Path


app = FastAPI()

torch.random.manual_seed(0)


# TODO: debug here, baseline_model folder
"""
Debug issue
"""


class InputData(BaseModel):
    text: str


@app.post("/score")
async def score(input_data: InputData):
    text = input_data.text

    print("### main_baseline.py: start run")

    model = AutoModelForCausalLM.from_pretrained(
        "/mnt/oga_models/hf_version/",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("/mnt/oga_models/hf_version/")

    # Initialize the pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

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
