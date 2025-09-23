"""Deterministic seeding utilities."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
