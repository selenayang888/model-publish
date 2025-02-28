# %%
from pprint import pprint
from typing import List, Dict, Any, Optional

import pandas as pd
import random
import json

from azure.ai.evaluation.simulator import AdversarialSimulator
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from app_target import ModelEndpoints
from azure.ai.evaluation.simulator import AdversarialScenario
from pathlib import Path

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
credential = DefaultAzureCredential()


# %%
async def callback(
    messages: List[Dict],
    stream: bool = False,
    session_state: Any = None,
) -> dict:
    query = messages["messages"][0]["content"]
    context = None

    # Add file contents for summarization or re-write
    if "file_content" in messages["template_parameters"]:
        query += messages["template_parameters"]["file_content"]

    # Call your own endpoint and pass your query as input. Make sure to handle your function_call_to_your_endpoint's error responses.
    model = "onnx-model"
    target = ModelEndpoints(env_var, model)
    response = target(query)["response"]

    # Format responses in OpenAI message protocol
    formatted_response = {
        "content": response,
        "role": "assistant",
        "context": {},
    }

    messages["messages"].append(formatted_response)
    return {
        "messages": messages["messages"],
        "stream": stream,
        "session_state": session_state,
    }


# %%
async def async_main_summarization(baseline_only=False):

    scenario = AdversarialScenario.ADVERSARIAL_SUMMARIZATION
    adversarial_simulator = AdversarialSimulator(
        azure_ai_project=azure_ai_project, credential=credential
    )

    outputs = await adversarial_simulator(
        scenario=scenario,  # required adversarial scenario to simulate
        target=callback,  # callback function to simulate against
        max_conversation_turns=1,  # optional, applicable only to conversation scenario
        max_simulation_results=250,  # optional
    )

    # By default simulator outputs json, use the following helper function to convert to QA pairs in jsonl format
    # outputs.to_eval_qa_json_lines()
    # print(outputs.to_eval_qa_json_lines())

    # %%

    with Path.open("outputs_safety.jsonl", "w") as f:
        f.write(outputs.to_eval_qr_json_lines())

    # %%
    filepath = "outputs_safety.jsonl"
    df = pd.read_json(filepath, lines=True)
    # print(df.head())

    content_safety_evaluator = ContentSafetyEvaluator(
        azure_ai_project=azure_ai_project, credential=credential
    )

    models = [
        "model",
    ]

    path = str(pathlib.Path(pathlib.Path.cwd())) + "/outputs_safety.jsonl"

    for model in models:
        randomNum = random.randint(1111, 9999)
        results = evaluate(
            evaluation_name="Eval-Run-" + str(randomNum) + "-" + model.title(),
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
    # pprint(results)

    # %%
    pd.DataFrame(results["rows"])

    # %%
    pprint(results["metrics"])

    json_result = json.dumps(results, indent=4)

    if baseline_only:
        with Path.open("/baseline_model/rai_summarization_result.json", "w") as f:
            f.write(json_result)
    else:
        with Path.open("/model/rai_summarization_result.json", "w") as f:
            f.write(json_result)
