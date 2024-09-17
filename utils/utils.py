import time


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