from evaluate_models_target_ground import async_main_ground

import asyncio

loop = asyncio.get_event_loop()

# # Test the groundness
try:
    loop.run_until_complete(async_main_ground())
except Exception as e:
    pass
