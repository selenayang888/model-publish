from pprint import pprint
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import pathlib
import random
import json
from azure.identity import AzureCliCredential
from rai_config import EVALUATOR_CLASSES
from consts import get_logger

credential = AzureCliCredential()


logging = get_logger(__name__)


async def async_main_evaluator(rai_target_name: str, baseline_only=False):
    """
    Run the RAI evaluator for the specified target.
    Args:
        rai_target_name (str): The name of the RAI target to evaluate.
                                options are: ip, upia, grounding, eci, safety
        baseline_only (bool): Whether to pytorch model or onnx model.
    """

    logging.info("Start RAI evaluation for %s", rai_target_name)

    path = str(pathlib.Path(pathlib.Path.cwd())) + f"/outputs_{rai_target_name}.jsonl"

    Evaluator = EVALUATOR_CLASSES["rai_target_name"]
    eci_evaluator = Evaluator(azure_ai_project=azure_ai_project, credential=credential)

    randomNum = random.randint(1111, 9999)
    results = evaluate(
        evaluation_name="Eval-Run-" + str(randomNum) + "-" + model.title(),
        data=path,
        evaluators={
            f"{rai_target_name}": Evaluator,
        },
        evaluator_config={
            f"{rai_target_name}": {
                "query": "${data.query}",
                "response": "${data.response}",
            },
        },
        # azure_ai_project=azure_ai_project,  # optional to store the evaluation results in Azure AI Studio
    )

    pprint(results["metrics"])

    json_result = json.dumps(results, indent=4)

    if baseline_only:
        with Path.open("/baseline_model/rai_eci_result.json", "w") as f:
            f.write(json_result)
    else:
        with Path.open("/model/rai_eci_result.json", "w") as f:
            f.write(json_result)

    logging.info("Finished RAI evaluation for %s", rai_target_name)
