
from evaluate_models_target_safety import async_main_safety
import asyncio

loop = asyncio.get_event_loop()

# # Test the jailbreaking

try:
    loop.run_until_complete(async_main_safety())
except Exception as e:
    pass
