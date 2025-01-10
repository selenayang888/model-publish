import subprocess
import time
import sys
from evaluate_models_target_ip import async_main
from evaluate_models_target_eci import async_main_eci
from evaluate_models_target_jailbreak import async_main_jailbreak
from evaluate_models_target_ground import async_main_ground

import os
print("### ENV in docker_main.py")
print(os.environ)

import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# step.1 - launch "uvicorn main:app --reload" in shell
uvicorn_proc = subprocess.Popen(["uvicorn main:app --reload"], shell=True)
print(f'### onnx model server started')

# step.2 - wait for the server ready
time.sleep(20)

# step.3 - call the endpoint and do validation
print(" ### Start model endpoint :)")

loop = asyncio.get_event_loop()
loop.run_until_complete(async_main())
loop.run_until_complete(async_main_eci())
loop.run_until_complete(async_main_jailbreak())
loop.run_until_complete(async_main_ground())



# step.last - send SIGTERM to server and wait for it to exit
print('### before oskill.')
import signal
os.kill(-uvicorn_proc.pid, signal.SIGKILL)
print('### after oskill')
exit_code = uvicorn_proc.wait()
print(f'### process exit for onnx model server: {exit_code}')

time.sleep(10)

# run baseline model with RAI evaluation
uvicorn_proc = subprocess.Popen(["uvicorn main_pytorch:app --reload"], shell=True)

print(f'### pytorch server started')

time.sleep(30)


print(" ### Start baseline-model endpoint :)")

loop = asyncio.get_event_loop()
loop.run_until_complete(async_main(baseline_only=True))
loop.run_until_complete(async_main_eci(baseline_only=True))
loop.run_until_complete(async_main_jailbreak(baseline_only=True))
loop.run_until_complete(async_main_ground(baseline_only=True))



print('### before oskill.')
os.kill(-uvicorn_proc.pid, signal.SIGKILL)
print('### after oskill')
exit_code = uvicorn_proc.wait()
print(f'### process exit for pytorch server: {exit_code}')

