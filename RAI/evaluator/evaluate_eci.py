# %%
from pprint import pprint
from pathlib import Path
from typing import List, Dict, Any, Optional


import pandas as pd
import pathlib
import random
import json

from azure.ai.evaluation.simulator import AdversarialSimulator
from azure.identity import AzureCliCredential
from app_target import ModelEndpoints

# ECI
from azure.ai.evaluation.simulator import AdversarialScenario
from azure.ai.evaluation.simulator._adversarial_scenario import (
    _UnstableAdversarialScenario,
)

from azure.ai.evaluation import evaluate
from azure.ai.evaluation._evaluators._eci._eci import ECIEvaluator

credential = AzureCliCredential()

async def callback(
    messages: List[Dict],
    stream: bool = False,
    session_state: Any = None,
) -> dict:

    query = messages["messages"][0]["content"]
    context = None

    if "file_content" in messages["template_parameters"]:
        query += messages["template_parameters"]["file_content"]

    target = ModelEndpoints()

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


async def async_main_eci(baseline_only=False):



    eci_evaluator = ECIEvaluator(
        azure_ai_project=azure_ai_project, credential=credential
    )

    path = str(pathlib.Path(pathlib.Path.cwd())) + "/outputs_eci.jsonl"

    randomNum = random.randint(1111, 9999)
    results = evaluate(
        evaluation_name="Eval-Run-" + str(randomNum) + "-" + model.title(),
        data=path,
        evaluators={
            "eci": eci_evaluator,
        },
        evaluator_config={
            "eci": {"query": "${data.query}", "response": "${data.response}"},
        },
        # azure_ai_project=azure_ai_project,  # optional to store the evaluation results in Azure AI Studio
    )

    pprint(results)

    pd.DataFrame(results["rows"])

    pprint(results["metrics"])

    json_result = json.dumps(results, indent=4)

    if baseline_only:
        with Path.open("/baseline_model/rai_eci_result.json", "w") as f:
            f.write(json_result)
    else:
        with Path.open("/model/rai_eci_result.json", "w") as f:
            f.write(json_result)
