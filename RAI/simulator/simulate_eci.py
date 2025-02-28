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

    scenario = _UnstableAdversarialScenario.ECI
    adversarial_simulator = AdversarialSimulator(
        azure_ai_project=azure_ai_project, credential=credential
    )

    outputs = await adversarial_simulator(
        scenario=scenario,  # required adversarial scenario to simulate
        target=callback,  # callback function to simulate against
        max_conversation_turns=1,  # optional, applicable only to conversation scenario
        max_simulation_results=250,  # optional
    )

    with Path.open("outputs_eci.jsonl", "w") as f:
        f.write(outputs.to_eval_qr_json_lines())