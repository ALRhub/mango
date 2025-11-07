import sys
from typing import Union, List, Optional

import h5py
import numpy as np
import torch.nn

from mango.dataset.util.graph_input_output_util import get_deformable_mask


def hdf5_group_to_dict(group):
    """
    Convert an HDF5 group of datasets to a dictionary of NumPy arrays.

    Parameters:
    - group (h5py.Group): The HDF5 group containing datasets.

    Returns:
    - dict: A dictionary where keys are dataset names and values are NumPy arrays.
    """
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            shape = item.shape
            if len(shape) != 0:
                result[key] = torch.from_numpy(item[:])
            else:
                # scalar dataset
                result[key] = torch.from_numpy(np.array(item))
            if result[key].dtype == torch.float64:
                result[key] = result[key].to(torch.float32)
        elif isinstance(item, h5py.Group):
            # If it's a nested group, recursively process it
            result[key] = hdf5_group_to_dict(item)
    return result


def convert_history_pos_to_history_vel_features(history_pos, current_pos):
    """
    Convert a history of positions to a history of velocities by calculating the difference between consecutive positions.

    Parameters:
    - history_pos (torch.Tensor): A tensor of shape (T, N, D) containing the history of positions. (oldest to newest)
    - current_pos (torch.Tensor): A tensor of shape (N, D) containing the current positions.

    Returns:
    - torch.Tensor: A tensor of shape (N, D * T) containing the history of velocities.
    """
    if history_pos.shape[0] == 0:
        return torch.zeros((current_pos.shape[0], 0))
    history_pos = torch.cat([history_pos, current_pos.unsqueeze(0)], dim=0)
    history_vel = history_pos[1:, :] - history_pos[:-1, :]
    # shape (T, N , D)
    # flatten
    history_vel = history_vel.view(history_vel.shape[1], -1)
    return history_vel

def add_noise(noise_scale, pos, history_pos, node_id, deformable_ids):
    if noise_scale == 0:
        return pos, history_pos
    # create  the deformable mask by hand since no batch is created yet
    deformable_mask = (node_id[:, None] == deformable_ids.view(1, -1)).any(dim=1)
    total_history_length = len(history_pos) + 1  # current position as well
    history_noise_scale = noise_scale / torch.sqrt(torch.tensor(total_history_length, dtype=torch.float32))
    len_deformable_nodes = torch.sum(deformable_mask)
    num_pos_features = pos.shape[1]
    noise = torch.randn(total_history_length, len_deformable_nodes, num_pos_features) * history_noise_scale
    # cumulative sum -> random walk
    noise = noise.cumsum(dim=0)
    history_noise = noise[:-1, :, :]
    current_noise = noise[-1, :, :]
    history_pos[:, deformable_mask, :] += history_noise
    pos[deformable_mask] += current_noise
    return pos, history_pos
