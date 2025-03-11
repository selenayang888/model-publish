import os

# For Simulation/Evaluation
os.environ["AZURE_SUBSCRIPTION_ID"] = "3905431d-c062-4c17-8fd9-c51f89f334c4"
os.environ["AZURE_RESOURCE_GROUP"] = "yangselenaai"
os.environ["AZURE_PROJECT_NAME"] = "azure_ai_studio_sdk"

# For LLM (Image Understanding use case)
os.environ["AZURE_DEPLOYMENT_NAME"] = "gpt-4o-mini"
os.environ["AZURE_ENDPOINT"] = "https://ai-yangselenaai3739831789912690.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview"
os.environ["AZURE_API_VERSION"] = "2024-06-01"
os.environ["AZURE_API_KEY"] = "0152bce79cdf40adab70375917f4b8ec"

# For LLM Dall-e-3 (Image generation)
os.environ["AZURE_DEPLOYMENT_NAME_DALLE"] = ""
os.environ["AZURE_ENDPOINT_DALLE"] = ""
os.environ["AZURE_API_VERSION_DALLE"] = ""

from azure.identity import AzureCliCredential
azure_cred = AzureCliCredential()
project_scope = {
    "subscription_id": os.environ.get("AZURE_SUBSCRIPTION_ID"),
    "resource_group_name": os.environ.get("AZURE_RESOURCE_GROUP"),
    "project_name": os.environ.get("AZURE_PROJECT_NAME"),
}

env_var = {
    "onnx-model": {
        "endpoint": "http://127.0.0.1:8000/score",
        "key": "",
    },
}

from pprint import pprint
import asyncio
import os
import json
import pandas as pd
from pathlib import Path

from openai import AzureOpenAI 
from typing import Any, Dict, List, Optional
from app_target_vision import ModelEndpoints

from azure.ai.evaluation.simulator import AdversarialScenario, AdversarialSimulator
from azure.ai.evaluation.simulator._adversarial_scenario import _UnstableAdversarialScenario


from azure.ai.evaluation import (
    ContentSafetyEvaluator,
    evaluate,
)


async def callback(
    messages: List[Dict],
    stream: bool = False,
    session_state: Any = None,
    context: Optional[Dict[str, Any]] = None,
) -> dict:
    image_understanding_prompt = messages["messages"][0]["content"]
    # call your endpoint or ai application here
    model = "onnx-model"
    target=ModelEndpoints(env_var, model)
    content = target(image_understanding_prompt)["response"]
    #content = call_gen_ai_application_or_llm(image_understanding_prompt, "You are an AI assistant who can describe images.")
    formatted_response = {
        "content": content, 
        "role": "assistant"
    }
    messages["messages"].append(formatted_response)
    return {
        "messages": messages["messages"],
        "stream": stream,
        "session_state": session_state,
        "context": context,
    }
            

async def main():

    simulator = AdversarialSimulator(azure_ai_project=project_scope, credential=azure_cred)

    simulator_output = await simulator(
            scenario=_UnstableAdversarialScenario.ADVERSARIAL_IMAGE_MULTIMODAL,
            max_conversation_turns=1,
            max_simulation_results=200,
            target=callback,
            api_call_retry_limit=3,
            api_call_retry_sleep_sec=1,
            api_call_delay_sec=20,
            concurrent_async_task=1,
        )

    pprint(simulator_output)
    
    file_name = "eval_vision_safety_ruiren.jsonl"

    # Write the output to the file
    with open(file_name, "w") as file:

        temp = []

        for conversation in simulator_output:
            
            if len(conversation["messages"]) > 1500:

                file.writelines([json.dumps({"conversation":{"messages": conversation["messages"][:1500]}}) + '\n'])
            else:

                file.writelines([json.dumps({"conversation":{"messages": conversation["messages"]}}) + '\n'])


    # Evaluator simulator output
    safety_eval = ContentSafetyEvaluator(azure_cred, project_scope)
    # run the evaluation
    eval_output = evaluate(
        data=file_name,
        evaluation_name="sim_image_understanding_safety_eval",
        #azure_ai_project=project_scope,
        evaluators={"safety": safety_eval},
    )

    row_result_df = pd.DataFrame(eval_output["rows"])
    metrics = eval_output["metrics"]
    pprint(row_result_df)
    pprint(metrics)
    
    json_result = json.dumps(eval_output, indent=4)


    with Path.open("./rai_vision_safety_result_new_ruiren.json", "w") as f:
        f.write(json_result)


asyncio.run(main())