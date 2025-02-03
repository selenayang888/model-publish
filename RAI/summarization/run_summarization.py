import asyncio
from evaluate_models_target_summarization import async_main_summarization


loop = asyncio.get_event_loop()


# Test the eci
try:
    loop.run_until_complete(async_main_summarization())
except Exception as e:
    pass