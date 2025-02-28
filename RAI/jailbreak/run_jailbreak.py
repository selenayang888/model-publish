
from evaluate_models_target_jailbreak import async_main_jailbreak

import os


import asyncio

loop = asyncio.get_event_loop()


# # Test the jailbreaking

try:
    loop.run_until_complete(async_main_jailbreak())
except Exception as e:
    pass
