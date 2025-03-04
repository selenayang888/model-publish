from pprint import pprint
from typing import List, Dict, Any, Optional
import pandas as pd
import random
import json
from azure.identity import AzureCliCredential, get_bearer_token_provider
from app_target import ModelEndpoints
from pathlib import Path
import pathlib
from callback_func import callback
from rai_config import SIMULATOR_CONFIG
from consts import get_logger


credential = AzureCliCredential()


logging = get_logger(__name__)


async def async_main_summarization(rai_target_name: str, baseline_only=False):
    """
    Run the RAI evaluator for the specified target.
    Args:
        rai_target_name (str): The name of the RAI target to evaluate.
                                options are: ip, upia, grounding, eci, safety
        baseline_only (bool): Whether to pytorch model or onnx model.
    """

    logging.info("Start RAI simulation for %s", rai_target_name)

    azure_ai_project = os.environ["azure_ai_project"]

    scenario = SIMULATOR_CONFIG[rai_target_name]["scenario"]
    max_results = SIMULATOR_CONFIG[rai_target_name]["max_results"]
    simulator = SIMULATOR_CONFIG[rai_target_name]["simulator_class"]
    max_conversation_turns = SIMULATOR_CONFIG[rai_target_name]["max_conversation_turns"]

    adversarial_simulator = simulator(
        azure_ai_project=azure_ai_project,
        credential=credential,
    )

    outputs = await adversarial_simulator(
        scenario=scenario,  # required adversarial scenario to simulate
        target=callback,  # callback function to simulate against
        max_conversation_turns=max_conversation_turns,  # optional, applicable only to conversation scenario
        max_simulation_results=max_results,  # optional
    )

    with Path.open(f"outputs_{rai_target_name}.jsonl", "w") as f:
        f.write(outputs.to_eval_qr_json_lines())

    logging.info("Finished RAI simulation for %s", rai_target_name)
