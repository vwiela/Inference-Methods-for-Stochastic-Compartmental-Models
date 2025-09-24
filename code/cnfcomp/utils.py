import time
import logging
import numpy as np

def time_counter():
    return time.process_time()

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def inverse_sigmoid(y):
    # Avoid division by zero or overflow
    y = np.clip(y, 1e-10, 1 - 1e-10)
    return np.log(y/(1-y))

def unconstrain(x, lower, upper):
    scaled = (x - lower) / (upper - lower)
    return inverse_sigmoid(scaled)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
"""
def unconstrain2(x, lower, upper):
    x_scaled = 2 * (x - lower) / (upper - lower) - 1
    return np.arctanh(x_scaled)
"""
def constrain(x, lower, upper):
    return (upper - lower) * sigmoid(x) + lower

"""
def constrain2(x, lower, upper):
    x_scaled = np.tanh(x)
    return 0.5 * (x_scaled + 1) * (upper - lower) + lower
    """