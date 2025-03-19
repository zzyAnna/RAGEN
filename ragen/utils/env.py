# env_utils.py
import random
import logging
import numpy as np
import torch
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from contextlib import contextmanager
import os

@contextmanager
def all_seed(seed):
    random_state = random.getstate()
    np_random_state = np.random.get_state()

    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)