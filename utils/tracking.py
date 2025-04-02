# utils/tracking.py
import time
import logging

def time_tracker(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(f"Entering {func.__name__}()")
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        logging.info(f"Exiting {func.__name__}() -- Elapsed: {elapsed:.2f}s")
        return result
    return wrapper
