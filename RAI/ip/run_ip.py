import subprocess
import time
import sys
from evaluate_models_target_ip import async_main
from evaluate_models_target_eci import async_main_eci
from evaluate_models_target_jailbreak import async_main_jailbreak
from evaluate_models_target_ground import async_main_ground
from evaluate_models_target_safety import async_main_safety
from evaluate_models_target_summarization import async_main_summarization

import os




import asyncio

loop = asyncio.get_event_loop()


# Test the IP
try:
    loop.run_until_complete(async_main())
except Exception as e:
    pass

