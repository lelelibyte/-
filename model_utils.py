"""Shared utilities for model construction and device management."""

import torch
from torch import nn


def get_device() -> torch.device:
    """Return the best available device (CUDA when present)."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(input_dim: int) -> nn.Sequential:
    """Create the feedforward classifier used across train/eval/predict."""

    return nn.Sequential(
        nn.Linear(input_dim, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
        nn.ReLU(),
        nn.Linear(10, 2),
    )
