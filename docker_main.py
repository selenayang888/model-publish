import subprocess
import time
import sys
from evaluate_models_target_ip import async_main
from evaluate_models_target_eci import async_main_eci

import os
print("### ENV in docker_main.py")
print(os.environ)

# step.1 - launch "uvicorn main:app --reload" in shell
uvicorn_proc = subprocess.Popen(["uvicorn main:app --reload"], shell=True)

# step.2 - wait for the server ready
time.sleep(10)

# step.3 - call the endpoint and do validation
print(" ### Start model endpoint :)")

import asyncio
loop = asyncio.get_event_loop()
loop.run_until_complete(async_main())
loop.run_until_complete(async_main_eci())


# step.last - send SIGTERM to server and wait for it to exit
uvicorn_proc.send_signal(subprocess.signal.SIGTERM)
uvicorn_proc.wait()
#sys.exit(uvicorn_proc.returncode)
