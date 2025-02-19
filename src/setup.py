import torch
import random
import os


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int = 0) -> None:
    random.seed(seed)


def get_cpu_count() -> int:
    return os.cpu_count()
