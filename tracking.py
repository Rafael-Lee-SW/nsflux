# tracking.py
import time
import logging

# Configure logging however you like
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

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
