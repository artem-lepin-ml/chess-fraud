import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required for deterministic cuBLAS ops under torch.use_deterministic_algorithms.
    # Must be set before the first CUDA/cuBLAS call in the process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        # warn_only=True so ops without deterministic impl warn rather than raise.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
