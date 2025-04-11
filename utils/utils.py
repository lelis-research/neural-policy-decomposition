import sys
import os
import time
import logging


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} executed in {end_time - start_time} seconds")
        return result
    return wrapper

def get_ppo_model_file_name(tag="", **kwargs):
    file_name = f'binary/PPO' + \
        f'-{kwargs["problem"]}' + \
        f'-game-width{kwargs["game_width"]}' + \
        f'-hidden{kwargs["hidden_size"]}' + \
        f'{tag}_MODEL.pt'
        # f'-l1lambda{kwargs["l1_lambda"]}' + \
    return file_name

def get_logger(logger_name, log_level, log_path):
    # Logger configurations
    os.makedirs(log_path, exist_ok=True)
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level.upper())
    log_path += "/" # Making sure the file and dir are separated
    log_path += logger_name
    log_path = f"{log_path}_{str(int(time.time()))}.log"
    handler = logging.FileHandler(log_path, mode='w')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(file_format)
    console_handler.setFormatter(console_format)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    return logger, log_path

def logger_flush(logger):
    for handler in logger.handlers:
        handler.flush()