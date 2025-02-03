import asyncio
from evaluate_models_target_eci import async_main_eci


loop = asyncio.get_event_loop()


# Test the eci
try:
    loop.run_until_complete(async_main_eci())
except Exception as e:
    pass