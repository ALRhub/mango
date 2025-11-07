import torch

from mango.dataset.ml_datasets.sphere_cloth import SphereClothDataset
from mango.dataset.util.util import add_noise
from mango.util.own_types import ConfigDict


class StepSphereClothDataset(SphereClothDataset):
    def __init__(self, config: ConfigDict, transform=None, pre_transform=None, pre_filter=None, split="train"):
        super().__init__(config, transform, pre_transform, pre_filter, split)
        self.start_time_steps = self.traj_length - self.config.future_length

    def len(self):
        true_length = len(self.tasks) * self.task_size * self.start_time_steps
        if self.config.get("dataset_size", None) is None:
            return true_length
        else:
            return min(self.config.dataset_size, true_length)

    def get_pos(self, traj_dict, step_idx):
        mesh_pos = traj_dict["cloth_positions"][step_idx]
        collider_pos = traj_dict["sphere_positions"][step_idx]
        mesh_pos = self.normalization(mesh_pos)
        collider_pos = self.normalization(collider_pos)
        x = torch.cat((mesh_pos, collider_pos), dim=0)
        return x, mesh_pos, collider_pos

    def get(self, idx):
        all_traj_idx, step_idx = idx // self.start_time_steps, idx % self.start_time_steps
        task_idx, traj_idx = all_traj_idx // self.task_size, all_traj_idx % self.task_size
        task_dict = self.tasks[task_idx]
        traj_key = f"traj_{traj_idx:03d}"
        traj_dict = task_dict["trajs"][traj_key]
        x, mesh_pos, collider_pos = self.get_pos(traj_dict, step_idx)
        x_init, _, _ = self.get_pos(traj_dict, 0)

        # build node feature tensor
        h, h_description, _ = self.get_current_node_features(traj_dict, task_dict, x[None, None])
        node_id_int = torch.zeros(x.shape[0], dtype=torch.long)
        # noise everything, since we want to predict everything
        x, _ = add_noise(self.config.noise_scale, x, history_pos=torch.zeros((0, x.shape[0], x.shape[1])),
                         node_id=node_id_int, deformable_ids=torch.tensor([0]))
        next_pos, next_mesh_pos, next_collider_pos = self.get_pos(traj_dict, step_idx + 1)
        if step_idx == 0:
            if self.config.initial_vel == "zero":
                v = torch.zeros_like(x)
            elif self.config.initial_vel == "constant":
                v = next_pos - x
            else:
                raise ValueError(f"Unknown initial velocity type: {self.config.initial_vel}")
        else:
            prev_pos, _, _ = self.get_pos(traj_dict, step_idx - 1)
            v = x - prev_pos

        edge_indices = self.edge_indices
        edge_features = self.base_edge_features
        init_r_ij = self.get_initial_rel_features(x_init, edge_indices)
        edge_features = torch.cat([edge_features, init_r_ij], dim=-1)
        edge_feature_description = ["one_hot"] * 5 + ["init_r_ij"] * 2
        result = {
            "x": x,  # shape (num_nodes, world_dim)
            "v": v,  # shape (num_nodes, world_dim)
            "h": h,  # shape (num_nodes, node_feature_dim)
            "h_description": h_description,
            "edge_indices": edge_indices,  # shape (2, num_edges)
            "edge_features": edge_features,  # shape (num_edges, num_edge_features)
            "edge_feature_description": edge_feature_description,
            "y": next_pos,  # shape (num_nodes, world_dim) , we predict collider in this task
        }
        if self.config.return_type == "torch_geometric" or self.config.return_type == "torch_geometric_without_vel":
            # transform to torch geometric data
            result = self.transform_to_torch_geometric(result)
        elif self.config.return_type == "dict":
            pass
        else:
            raise ValueError(f"Unknown return type: {self.config.return_type}")
        return result
