import os
import os.path as osp
from typing import Union

import numpy as np
import torch

from mango.dataset.edges.edge_indices import get_edge_indices
from mango.dataset.ml_datasets.abstract_ml_dataset import AbstractMLDataset
from mango.util.own_types import ConfigDict
from mango.util.util import to_numpy


class PlanarBendingDataset(AbstractMLDataset):
    def __init__(self, config: ConfigDict, transform=None, pre_transform=None, pre_filter=None, split="train"):
        super().__init__(config, transform, pre_transform, pre_filter, split)
        assert self.config.time_subsampling == 1, "Time subsampling not supported for this dataset"
        # normalization
        self.normalization_factor = 140
        self.force_normalization_factor = 200
        self.boarder_indices = self.get_node_indices_on_boarder()
        # edge indices tensor
        faces = self.global_data["cells"]
        self.edge_indices = get_edge_indices(faces)

    def get_node_indices_on_boarder(self):
        # first step of first traj of first task
        nodes = np.array(self.tasks[0]["trajs"]["traj_000"]["mesh_pos"][0])
        # do it in numpy since code was already implemented like that
        x_min = np.min(nodes[:, 0])
        x_max = np.max(nodes[:, 0])
        y_min = np.min(nodes[:, 1])
        y_max = np.max(nodes[:, 1])
        # get the node with indices < size
        node_indices = \
            np.where((nodes[:, 0] == x_min) | (nodes[:, 0] == x_max) | (nodes[:, 1] == y_min) | (nodes[:, 1] == y_max))[
                0]
        return torch.tensor(node_indices)

    def get(self, task_idx):
        task_dict = self.tasks[task_idx]
        # all pos
        x = torch.stack([task_dict["trajs"][traj_key]["mesh_pos"] for traj_key in sorted(task_dict["trajs"].keys())])
        # normalization
        x = x / self.normalization_factor
        # build h tensor
        h = []
        for traj_key in sorted(task_dict["trajs"].keys()):
            traj_dict = task_dict["trajs"][traj_key]
            current_h, h_description, regression_features = self.get_current_node_features(traj_dict, task_dict)
            h.append(current_h)
        h = torch.stack(h)
        # velocity
        v = self.compute_velocity(x)
        edge_indices = self.edge_indices
        edge_features = torch.zeros(edge_indices.shape[1], 2)  # edge id: 0: node-node, 1: node-encoder
        edge_features[:, 0] = 1.0
        edge_features = edge_features[None, :, :].repeat(h.shape[0], 1, 1)
        init_r_ij = self.get_initial_rel_features(x, edge_indices)
        edge_features = torch.cat([edge_features, init_r_ij], dim=-1)
        edge_feature_description = ["one_hot"] * 2 + ["init_r_ij"] * 3
        context_trajs, target_trajs = self.get_context_target_trajs(x.shape[0])
        result = {
            "x": x,  # shape (num_trajs_in_task, traj_length, num_nodes, world_dim)
            "v": v,  # shape (num_trajs_in_task, traj_length, num_nodes, world_dim)
            "h": h,  # shape (num_trajs_in_task, num_nodes, node_feature_dim)
            "h_description": h_description,
            "edge_indices": edge_indices,  # shape (2, num_edges)
            "edge_features": edge_features,  # shape (num_trajs_in_task, num_edges, num_edge_features)
            "edge_feature_description": edge_feature_description,
            "context_trajs": context_trajs,  # shape (num_context_trajs)
            "target_trajs": target_trajs  # shape (num_target_trajs)

        }

        if self.config.regression:
            result["regression_features"] = regression_features
        return result

    def get_current_node_features(self, traj_dict, task_dict):
        h_description = ["one_hot", "one_hot"]
        init_pos = traj_dict["mesh_pos"][0]
        force_features = []
        for force_dict in traj_dict["params"].values():
            force_nodes = self.get_node_indices_with_force_influence(init_pos, force_dict["position"])
            force_feature = torch.zeros((init_pos.shape[0], 1))
            force_feature[force_nodes] = force_dict["direction"][2] / self.force_normalization_factor
            force_features.append(force_feature)
            h_description.append("force")
        # 2 node ids: normal mesh node: 0, output encoder node: 1
        node_id = torch.zeros(init_pos.shape[0], 2)
        node_id[:, 0] = 1.0
        force_features = torch.cat(force_features, dim=-1)
        boarder_features = torch.zeros(init_pos.shape[0], 1)
        boarder_features[self.boarder_indices] = 1.0
        current_h = torch.cat([node_id, force_features, boarder_features], dim=-1)
        regression_features = None
        h_description.append("boarder")
        if self.config.material_properties or self.config.regression:
            # add material proprties
            material_properties = torch.tensor([task_dict["params"]["youngs_modulus"]])
            # max youngs_modulus is 500
            material_properties = material_properties / 500
            if self.config.regression:
                regression_features = material_properties
            material_properties = material_properties.repeat(init_pos.shape[0], 1)
            if self.config.material_properties:
                current_h = torch.cat([current_h, material_properties], dim=-1)
                h_description.append("material_properties")
        return current_h, h_description, regression_features

    def get_node_indices_with_force_influence(self, nodes, force_pos):
        force_application = {
            "length": 20,
            "search_mode": "Cylinder"
        }
        if force_application["search_mode"] == "Cylinder":
            radius = force_application["length"] / 2
            distances = torch.linalg.norm(nodes - force_pos, axis=1)
            # get the node with indices < size
            node_indices = torch.where(distances < radius)[0]
            return node_indices

    def get_faces(self):
        faces = self.global_data["cells"]
        faces = {"deformable": to_numpy(faces)}
        return faces

    @property
    def traj_length(self):
        return 51
