import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_logs"
log_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

os.makedirs(log_path, exist_ok=True)

LOG_FILES_PATH = os.path.join(log_path, LOG_FILE)


logging.basicConfig(
    filename=LOG_FILES_PATH,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    level=logging.INFO,
    filemode='a')


