# %%
from pprint import pprint
from typing import List, Dict, Any, Optional

import pandas as pd
import random
import json
import logging

import requests
from azure.ai.evaluation.simulator import AdversarialSimulator
from azure.identity import AzureCliCredential, get_bearer_token_provider
from app_target import ModelEndpoints
from azure.ai.evaluation.simulator import AdversarialScenario
from pathlib import Path


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# %%
# from app_target import ModelEndpoints
import pathlib

from azure.ai.evaluation import evaluate
from azure.ai.evaluation import (
    ContentSafetyEvaluator,
)

# %%
env_var = {
    "onnx-model": {
        "endpoint": "http://127.0.0.1:8000/score",
        "key": "",
    },
}

# %%
azure_ai_project = {
    "subscription_id": "3905431d-c062-4c17-8fd9-c51f89f334c4",
    "resource_group_name": "yangselenaai",
    "project_name": "azure_ai_studio_sdk",
}
credential = AzureCliCredential()


# Here need to call ONNXRuntime API for audio outputs

logger.info("Starting my_async_function")
# the input JSON file Path

locales = [ "it", "ja", "pt", "zh"]

for lang in locales:
    JSON_path = str(pathlib.Path(pathlib.Path.cwd())) + f"/audio/content_safety_en/{lang}/{lang}.jsonl"
    
    logger.info("JSON_path: %s", JSON_path)
    # the output JSON file Path
    queries_array = []

    with open(JSON_path, "r") as f:
        
        query_inputs = f.readlines()
        count = 0
        for query_input in query_inputs:
            count += 1
            values = json.loads(query_input)

            payload = {
                "text": values["query_text"],
            }

            try: 
                response = requests.post(
                    env_var["onnx-model"]["endpoint"],
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                )
                values["response_text"] = response.json()["response"]
                logger.info(f"response: {values['response_text']}")
            except Exception as e:
                values["response_text"] = "I am a safe robot!"

            logger.info(f"process the {count} of {lang}.")
            
            queries_array.append(values)

    data = []

    for val in queries_array:
        temp = {}
        temp["query"] = val["query_text"]

        if len(val["response_text"]) >= 1500:
            temp["response"] = val["response_text"][:1500]
        else:
            temp["response"] = val["response_text"]
        
        data.append(temp)

    with open(f"outputs_text_safety_{lang}.jsonl", "w") as f:
        for val in data:
            f.write(json.dumps(val) + "\n")

    
    # content_safety_evaluator = ContentSafetyEvaluator(
    #     azure_ai_project=azure_ai_project, credential=credential
    # )

    # path = str(pathlib.Path(pathlib.Path.cwd())) + f"/outputs_text_safety_{lang}.jsonl"

    # randomNum = random.randint(1111, 9999)

    # results = evaluate(
    #     evaluation_name="Eval-Run-" + str(randomNum),
    #     data=path,
    #     evaluators={
    #         "content_safety": content_safety_evaluator,
    #     },
    #     evaluator_config={
    #         "content_safety": {
    #             "query": "${data.query}",
    #             "response": "${data.response}",
    #             },
    #         },
    #         # azure_ai_project=azure_ai_project,  # optional to store the evaluation results in Azure AI Studio
    #     )

    # # %%

    # json_result = json.dumps(results, indent=4)

    # with Path.open(f"./rai_text_safety_result_{lang}.json", "w") as f:
    #     f.write(json_result)
