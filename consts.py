import logging
from dotenv import load_dotenv

load_dotenv()

def get_logger(name: str):
    """
    Return logger instance with given name
    """

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    if not logger.hasHandlers():

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
            ),
        )
        logger.addHandler(console_handler)

    return logger
