"""
utils.py
--------
Helper functions for the battery SOH prediction pipeline.

Author: Prateek Gaur
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seeds for full reproducibility across numpy, torch, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Convert seconds to human-readable string — e.g. 125s → '2m 5s'."""
    minutes = int(seconds // 60)
    secs    = int(seconds % 60)
    return f"{minutes}m {secs}s" if minutes > 0 else f"{secs}s"
