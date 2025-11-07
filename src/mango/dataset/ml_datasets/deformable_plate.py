import numpy as np
import torch
from mango.dataset.edges.edge_indices import get_edge_indices
from mango.dataset.ml_datasets.abstract_ml_dataset import AbstractMLDataset
from mango.util.own_types import ConfigDict
from mango.util.util import to_numpy


class DeformablePlateDataset(AbstractMLDataset):
    def __init__(self, config: ConfigDict, transform=None, pre_transform=None, pre_filter=None, split="train"):
        super().__init__(config, transform, pre_transform, pre_filter, split)
        assert self.config.time_subsampling == 1, "Time subsampling not supported for this dataset"
        # edge indices tensor
        self.compute_edge_indices()

        # normalization
        self.min_x = torch.min(torch.tensor([-168.8057, -100.00]))
        self.max_x = torch.max(torch.tensor([[173.7962, 215.2644]]))

    def compute_edge_indices(self):
        faces = self.get_faces()
        # offset collider indices
        faces["collider"] = faces["collider"] + np.max(faces["deformable"]) + 1
        self.mesh_mesh_indices = get_edge_indices(faces["deformable"])
        self.col_col_indices = get_edge_indices(faces["collider"])
        top_mesh_indices = torch.arange(36, 45)
        lower_collider_indices = torch.tensor([124, 123, 125, 122, 126, 121, 127, 120, 128, 119, 129, 118, 130, 117,
                                               131, 116, 132, 115, 133, 114, 134, 113, 135, 112, 136, 111, 137, 110])
        self.mesh_col_indices = torch.cartesian_prod(top_mesh_indices, lower_collider_indices).T
        self.col_mesh_indices = torch.cartesian_prod(lower_collider_indices, top_mesh_indices).T
        self.edge_indices = torch.cat(
            [self.mesh_mesh_indices, self.col_col_indices, self.mesh_col_indices, self.col_mesh_indices], dim=-1)

        edge_features = torch.zeros(self.edge_indices.shape[1], 5)
        # edge id: 0: node-node, 1: collider-collider, 2: node-collider, 3: collider-node, 4: node-encoder
        edge_features[:self.mesh_mesh_indices.shape[1], 0] = 1.0
        edge_features[
        (self.mesh_mesh_indices).shape[1]:(self.mesh_mesh_indices).shape[1] + (self.col_col_indices).shape[1], 1] = 1.0
        edge_features[
        (self.mesh_mesh_indices).shape[1] + (self.col_col_indices).shape[1]:(self.mesh_mesh_indices).shape[1] +
                                                                            (self.col_col_indices).shape[1] +
                                                                            (self.mesh_col_indices).shape[1], 2] = 1.0
        edge_features[
        (self.mesh_mesh_indices).shape[1] + (self.col_col_indices).shape[1] + (self.mesh_col_indices).shape[1]:, 3] = 1.0
        self.base_edge_features = edge_features
        self.edge_id = torch.zeros(self.edge_indices.shape[1], dtype=torch.long)
        self.edge_id[:self.mesh_mesh_indices.shape[1]] = 0
        self.edge_id[self.mesh_mesh_indices.shape[1]:(self.mesh_mesh_indices).shape[1] + (self.col_col_indices).shape[1]] = 1
        self.edge_id[(self.mesh_mesh_indices).shape[1] + (self.col_col_indices).shape[1]:(self.mesh_mesh_indices).shape[1] +
                                                                            (self.col_col_indices).shape[1] +
                                                                            (self.mesh_col_indices).shape[1]] = 2
        self.edge_id[(self.mesh_mesh_indices).shape[1] + (self.col_col_indices).shape[1] + (self.mesh_col_indices).shape[1]:] = 3

    def normalization(self, x):
        return (x - self.min_x) / (self.max_x - self.min_x)

    def get(self, task_idx):
        task_dict = self.tasks[task_idx]
        # all pos
        mesh_pos = torch.stack(
            [task_dict["trajs"][traj_key]["mesh_pos"] for traj_key in sorted(task_dict["trajs"].keys())])
        collider_pos = torch.stack(
            [task_dict["trajs"][traj_key]["collider_pos"] for traj_key in sorted(task_dict["trajs"].keys())])
        # normalization
        mesh_pos = self.normalization(mesh_pos)
        collider_pos = self.normalization(collider_pos)
        x = torch.cat((mesh_pos, collider_pos), dim=2)
        # # add z dim since the mesh is 2D
        # x = torch.cat((x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], 1)), dim=-1)
        # build h tensor
        h = []
        for traj_key in sorted(task_dict["trajs"].keys()):
            traj_dict = task_dict["trajs"][traj_key]
            current_h, h_description, regression_features = self.get_current_node_features(traj_dict, task_dict, x)
            h.append(current_h)
        h = torch.stack(h)
        # velocity
        v = self.compute_velocity(x)
        edge_indices = self.edge_indices
        edge_features = self.base_edge_features
        edge_features = edge_features[None, :, :].repeat(h.shape[0], 1, 1)
        init_r_ij = self.get_initial_rel_features(x, edge_indices)
        edge_features = torch.cat([edge_features, init_r_ij], dim=-1)
        edge_feature_description = ["one_hot"] * 5 + ["init_r_ij"] * 2
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
            result["regression_features"] = regression_features.squeeze()
        return result

    def get_current_node_features(self, traj_dict, task_dict, x):
        h_description = ["one_hot", "one_hot", "one_hot"]
        init_pos = traj_dict["mesh_pos"][0]
        # 3 node ids: deformable node: 0, collider_node: 1 output encoder node: 2
        node_id = torch.zeros(x.shape[2], 3)
        node_id[0:len(init_pos), 0] = 1.0
        node_id[len(init_pos):, 1] = 1.0
        regression_features = None
        current_h = node_id
        if self.config.material_properties or self.config.regression:
            # add material proprties
            youngs_modulus = torch.tensor([task_dict["params"]["youngs_modulus"]])
            poisson_ratio = torch.tensor([task_dict["params"]["poisson_ratio"]])
            # max youngs_modulus is 10000 for harder tasks
            youngs_modulus = youngs_modulus / 10000
            # poisson ratio is already between -1 and 0.5, so enough normalized
            material_properties = torch.cat([youngs_modulus, poisson_ratio], dim=-1)
            if self.config.regression:
                regression_features = material_properties
            if self.config.material_properties:
                material_properties = material_properties.repeat(x.shape[2], 1)
                current_h = torch.cat([current_h, material_properties], dim=-1)
                h_description.append("youngs_modulus")
                h_description.append("poisson_ratio")
        return current_h, h_description, regression_features

    def get_faces(self):
        faces = self.global_data["cell_indices"]
        faces = {key: to_numpy(face) for key, face in faces.items()}
        # shift collider faces by number of deformable nodes
        faces["collider_faces"] = faces["collider_faces"]
        faces["collider"] = faces["collider_faces"]
        faces["deformable"] = faces["mesh_faces"]
        del faces["collider_faces"]
        del faces["mesh_faces"]
        return faces

    @property
    def traj_length(self):
        return 52
