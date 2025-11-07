from abc import ABC, abstractmethod

import h5py
import torch
from torch_geometric.data import Dataset, Data

from mango.dataset.edges.edge_features import add_distances_from_positions
from mango.dataset.util.graph_input_output_util import get_deformable_mask, get_collider_mask
from mango.dataset.util.util import hdf5_group_to_dict
from mango.util.own_types import ConfigDict


class AbstractMLDataset(Dataset, ABC):
    def __init__(self, config: ConfigDict, transform=None, pre_transform=None, pre_filter=None, split="train"):
        self.config = config
        self.split = split
        root = config.root
        self.tasks = []
        with h5py.File(root, "r") as hdf:
            splits = hdf["splits"]
            self._my_indices = splits[f"{self.split}_indices"][:]
            self.global_data = hdf5_group_to_dict(hdf["global_data"])
            for task_key in sorted(hdf.keys()):
                if task_key.startswith("task_"):
                    task_index = int(task_key[-3:])
                    if task_index in self._my_indices:
                        self.tasks.append(hdf5_group_to_dict(hdf[task_key]))
        self.random_traj_selection = config.random_traj_selection
        self._task_size = len(self.tasks[0]["trajs"])
        # standard training mode
        super().__init__(root, transform, pre_transform, pre_filter)

    def len(self):
        if self.config.get("dataset_size", None) is None:
            return len(self.tasks)
        else:
            return min(self.config.dataset_size, len(self.tasks))

    @abstractmethod
    def get(self, task_idx):
        raise NotImplementedError

    @abstractmethod
    def get_faces(self):
        raise NotImplementedError

    def get_initial_rel_features(self, x, edge_index):
        row, col = edge_index
        if len(x.shape) == 4:
            r_ij = x[:, 0, row, :] - x[:, 0, col, :]  # relative position
        elif len(x.shape) == 2:
            r_ij = x[row, :] - x[col, :]  # relative position
        else:
            raise ValueError(f"Unknown shape of x: {x.shape}")
        return r_ij

    def get_context_target_trajs(self, task_size):
        if self.random_traj_selection:
            context_size = torch.randint(self.config.min_context_size, self.config.max_context_size + 1, (1,)).item()
            target_size = torch.randint(self.config.min_target_size, self.config.max_target_size + 1, (1,)).item()
        else:
            context_size = self.config.context_size
            target_size = self.config.target_size
        assert context_size <= task_size, "Context size must be smaller or equal to task size"
        assert target_size <= task_size, "Target size must be smaller or equal to task size"

        if self.random_traj_selection:
            # Perform sampling without replacement
            context_trajs = torch.multinomial(torch.ones(task_size), context_size, replacement=False)
            target_trajs = torch.multinomial(torch.ones(task_size), target_size, replacement=False)
        else:
            context_trajs = torch.arange(context_size)
            target_trajs = torch.arange(target_size)
        return context_trajs, target_trajs

    def compute_velocity(self, x):
        v = x[:, 1:, :, :] - x[:, :-1, :, :]
        if self.config.initial_vel == "zero":
            # zero padding for step 0 velocity
            v = torch.cat((torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3])), v), dim=1)
        elif self.config.initial_vel == "constant":
            # constant padding for step 0 velocity: velocity is the same as the velocity from step 1
            v = torch.cat((v[:, 0:1, :, :], v), dim=1)
        else:
            raise ValueError(f"Unknown initial velocity type: {self.config.initial_vel}")
        return v


    def update(self, data: Data, predicted_def_pos, new_time_step, edge_features):
        """
        Used for step-based algorithms to update the eval data with the predicted deformable positions.
        :param data:
        :param predicted_def_pos:
        :param new_time_step:
        :param edge_features:
        :return:
        """
        new_pos = torch.zeros_like(data.pos)
        deformable_mask = get_deformable_mask(data)
        collider_mask = get_collider_mask(data)
        new_pos[deformable_mask] = predicted_def_pos
        new_collider_pos = data.traj_pos[new_time_step]
        new_pos[collider_mask] = new_collider_pos[collider_mask]

        # update history vel features
        current_vel = new_pos - data.pos

        history_mask = torch.tensor([desc == "velocity" for desc in data.x_description], device=data.x.device)
        if torch.any(history_mask):
            history_vel = data.x[:, history_mask]
            world_dim = predicted_def_pos.shape[-1]
            # remove oldest history entry
            history_vel = history_vel[:, world_dim:]
            # add current vel to history
            history_vel = torch.cat([history_vel, current_vel], dim=-1)
            # replace x history features
            data.x[:, history_mask] = history_vel

        # update position
        data.pos = new_pos
        # remove edge_distances
        data.edge_attr = edge_features
        # add distances
        data = add_distances_from_positions(data, add_euclidian_distance=True)
        return data

    @property
    def traj_length(self):
        raise NotImplementedError

    @property
    def task_size(self):
        return self._task_size
