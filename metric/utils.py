# -*- coding: utf-8 -*-

import math
from dataclasses import asdict, is_dataclass

def safe_div(a, b, default=0.0):
    return a / b if b else default

def running_mean(prev_mean, prev_step, new_val):
    step = prev_step + 1
    mean = prev_mean + (new_val - prev_mean) / step
    return mean, step

def dataclass_to_dict(x):
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return {k: dataclass_to_dict(v) for k, v in x.items()}
    return x
