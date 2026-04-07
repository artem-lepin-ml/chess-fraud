import torch


def get_device() -> torch.device:
    """Return cuda if available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
