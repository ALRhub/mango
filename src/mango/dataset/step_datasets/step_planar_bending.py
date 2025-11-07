import torch
from torch_geometric.data import Data

from mango.dataset.edges.edge_features import add_distances_from_positions
from mango.dataset.ml_datasets.planar_bending import PlanarBendingDataset
from mango.dataset.util.util import hdf5_group_to_dict, add_noise
from mango.util.own_types import ConfigDict


class StepPlanarBendingDataset(PlanarBendingDataset):
    def __init__(self, config: ConfigDict, transform=None, pre_transform=None, pre_filter=None, split="train"):
        super().__init__(config, transform, pre_transform, pre_filter, split)
        self.start_time_steps = self.traj_length - self.config.future_length

    def len(self):
        true_length = len(self.tasks) * self.task_size * self.start_time_steps
        if self.config.get("dataset_size", None) is None:
            return true_length
        else:
            return min(self.config.dataset_size, true_length)

    def get(self, idx):
        all_traj_idx, step_idx = idx // self.start_time_steps, idx % self.start_time_steps
        task_idx, traj_idx = all_traj_idx // self.task_size, all_traj_idx % self.task_size
        task_dict = self.tasks[task_idx]
        traj_key = f"traj_{traj_idx:03d}"
        traj_dict = task_dict["trajs"][traj_key]
        # current pos
        x = traj_dict["mesh_pos"][step_idx]
        # init pos
        x_init = traj_dict["mesh_pos"][0]
        # normalization
        x = x / self.normalization_factor
        x_init = x_init / self.normalization_factor

        # 2 node ids: normal mesh node: 0, output encoder node: 1
        node_id = torch.zeros(x.shape[0], 2)
        node_id[:, 0] = 1.0
        node_id_int = torch.zeros(x.shape[0], dtype=torch.long)

        # add noise
        x, _ = add_noise(self.config.noise_scale, x, history_pos=torch.zeros((0, x.shape[0], x.shape[1])),
                         node_id=node_id_int, deformable_ids=torch.tensor([0]))

        next_pos = traj_dict["mesh_pos"][step_idx + 1]
        next_pos = next_pos / self.normalization_factor

        # build node feature tensor
        h, h_description, _ = self.get_current_node_features(traj_dict, task_dict)

        # velocity, same constant padding as in the ml dataset. In general, ablate this
        if step_idx == 0:
            if self.config.initial_vel == "zero":
                v = torch.zeros_like(x)
            elif self.config.initial_vel == "constant":
                v = next_pos - x
            else:
                raise ValueError(f"Unknown initial velocity type: {self.config.initial_vel}")
        else:
            prev_pos = traj_dict["mesh_pos"][step_idx - 1]
            prev_pos = prev_pos / self.normalization_factor
            v = x - prev_pos

        edge_indices = self.edge_indices
        edge_features = torch.zeros(edge_indices.shape[1], 2)  # edge id: 0: node-node, 1: node-encoder
        edge_features[:, 0] = 1.0
        init_r_ij = self.get_initial_rel_features(x_init, edge_indices)
        edge_features = torch.cat([edge_features, init_r_ij], dim=-1)
        edge_feature_description = ["one_hot"] * 2 + ["init_r_ij"] * 3
        result = {
            "x": x,  # shape (num_nodes, world_dim)
            "v": v,  # shape (num_nodes, world_dim)
            "h": h,  # shape (num_nodes, node_feature_dim)
            "h_description": h_description,
            "edge_indices": edge_indices,  # shape (2, num_edges)
            "edge_features": edge_features,  # shape (num_edges, num_edge_features)
            "edge_feature_description": edge_feature_description,
            "y": next_pos,  # shape (num_nodes, world_dim)
        }
        if self.config.return_type == "torch_geometric" or self.config.return_type == "torch_geometric_without_vel":
            # transform to torch geometric data
            result = self.transform_to_torch_geometric(result)
        elif self.config.return_type == "dict":
            pass
        else:
            raise ValueError(f"Unknown return type: {self.config.return_type}")
        return result

    def transform_to_torch_geometric(self, result, ):
        pos, v, h, h_description, edge_indices, edge_features, y = (result["x"], result["v"], result["h"], \
            result["h_description"], result["edge_indices"], \
            result["edge_features"], result["y"])

        # build node features (x tensor)
        if self.config.return_type == "torch_geometric":
            x = torch.cat([h, v], dim=-1)
            x_description = h_description + ["velocity"] * v.shape[-1]
        elif self.config.return_type == "torch_geometric_without_vel":
            x = h
            x_description = h_description
        else:
            raise ValueError(f"Unknown return type: {self.config.return_type}")
        node_id = torch.zeros(x.shape[0], dtype=torch.long)
        node_id_dict = {0: "deformable"}
        edge_id = torch.zeros(edge_indices.shape[1], dtype=torch.long)
        deformable_ids = torch.tensor([0])
        collider_ids = torch.tensor([])
        data = Data(
            x=x,
            y=y,
            pos=pos,
            node_id=node_id,
            node_id_dict=node_id_dict,
            x_description=x_description,
            edge_index=edge_indices.to(torch.int64),
            edge_attr=edge_features,
            edge_id=edge_id,
            deformable_ids=deformable_ids,
            collider_ids=collider_ids,
        )
        # put pos into edge attr
        data = add_distances_from_positions(data, add_euclidian_distance=True)
        return data
