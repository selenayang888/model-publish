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


logger.info("Starting my_async_function")
# the input JSON file Path

locales = [ "zh"]

for lang in locales:
    JSON_path = str(pathlib.Path(pathlib.Path.cwd())) + f"/audio/content_safety_en/{lang}/{lang}.jsonl"
    
    logger.info("JSON_path: %s", JSON_path)
    # the output JSON file Path

    path = str(pathlib.Path(pathlib.Path.cwd())) + f"/outputs_text_safety_{lang}.jsonl"

    content_safety_evaluator = ContentSafetyEvaluator(
        azure_ai_project=azure_ai_project, credential=credential
    )

    randomNum = random.randint(1111, 9999)

    results = evaluate(
        evaluation_name="Eval-Run-" + str(randomNum),
        data=path,
        evaluators={
            "content_safety": content_safety_evaluator,
        },
        evaluator_config={
            "content_safety": {
                "query": "${data.query}",
                "response": "${data.response}",
            },
        },
        # azure_ai_project=azure_ai_project,  # optional to store the evaluation results in Azure AI Studio
    )

    # %%

    json_result = json.dumps(results, indent=4)

    with Path.open(f"./rai_text_safety_result_{lang}.json", "w") as f:
        f.write(json_result)
