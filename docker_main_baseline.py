import subprocess
import time
import sys
from evaluate_models_target_ip import async_main
from evaluate_models_target_jailbreak import async_main_jailbreak
from evaluate_models_target_ground import async_main_ground
from evaluate_models_target_safety import async_main_safety
from evaluate_models_target_summarization import async_main_summarization

import os

print("### ENV in docker_main_baseline.py")
print(os.environ)


# run baseline model with RAI evaluation
uvicorn_proc = subprocess.Popen(["uvicorn main_pytorch:app --reload"], shell=True)

print(" ### Start baseline-model endpoint :)")


time.sleep(70)


import asyncio

loop = asyncio.get_event_loop()

# loop.run_until_complete(async_main(baseline_only=True))
# loop.run_until_complete(async_main_eci(baseline_only=True))
# loop.run_until_complete(async_main_jailbreak(baseline_only=True))
loop.run_until_complete(async_main_ground(baseline_only=True))
# loop.run_until_complete(async_main_safety(baseline_only=True))
# loop.run_until_complete(async_main_summarization(baseline_only=True))


uvicorn_proc.send_signal(subprocess.signal.SIGTERM)
uvicorn_proc.wait()
