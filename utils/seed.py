"""Global seed utility for reproducible experiments."""

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set random seed for Python, NumPy, and all supported libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
