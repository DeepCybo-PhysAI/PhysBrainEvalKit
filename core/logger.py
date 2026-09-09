import os
import time
import logging


def setup_logging(task_name, model_name):
    """Setup logging with both console and file handlers"""
    current_time = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs('logs', exist_ok=True)
    log_file_name = f"logs/{task_name}_{model_name}_{current_time}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_name)
        ]
    )
    logger = logging.getLogger(f"{task_name}_{model_name}")
    return log_file_name

