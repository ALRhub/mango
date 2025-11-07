from typing import Dict

import numpy as np
import torch


def to_numpy(dict_or_tensor: torch.Tensor | Dict[str, torch.Tensor]) -> np.array:
    """
    Converts a tensor to a numpy array
    """
    if isinstance(dict_or_tensor, dict):
        return {key: to_numpy(value) for key, value in dict_or_tensor.items()}
    else:
        return dict_or_tensor.detach().cpu().numpy()
