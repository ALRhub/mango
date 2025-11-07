import numpy as np
import torch
from matplotlib import pyplot as plt

from mango.dataset.edges.edge_indices import get_edge_indices
from mango.dataset.ml_datasets.abstract_ml_dataset import AbstractMLDataset
from mango.util.own_types import ConfigDict
from mango.util.util import to_numpy


class SphereClothDataset(AbstractMLDataset):
    def __init__(self, config: ConfigDict, transform=None, pre_transform=None, pre_filter=None, split="train"):
        super().__init__(config, transform, pre_transform, pre_filter, split)
        assert self.config.time_subsampling == 1, "Time subsampling not supported for this dataset"
        # edge indices tensor
        self.compute_edge_indices()

        self.border_indices = torch.tensor([0, 19, 380, 399])

        # normalization
        # TODO: Update this normalization, it for now is just for the toy dataset
        self.min_x = torch.min(torch.tensor([-2.1, -2.1, 0.5]))
        self.max_x = torch.max(torch.tensor([[2.1, 2.1, 11.0]]))

    def compute_edge_indices(self):
        faces = self.get_faces()
        # offset collider indices already in get_faces
        # faces["collider"] = faces["collider"] + np.max(faces["deformable"]) + 1
        self.mesh_mesh_indices = get_edge_indices(faces["deformable"])
        self.col_col_indices = get_edge_indices(faces["collider"])

        # Plot 3d points
        mesh = self.tasks[0]["trajs"]["traj_000"]["cloth_positions"][0]
        sphere = self.tasks[0]["trajs"]["traj_000"]["sphere_positions"][0]

        inner_mesh_indices = []
        for row in [120, 160, 200, 240, 280]:
            for col in [6, 8, 10, 12, 14]:
                inner_mesh_indices.append(row + col)
        inner_mesh_indices = torch.tensor(inner_mesh_indices)
        sphere_indices = torch.arange(len(mesh), len(mesh) + len(sphere), dtype=torch.long)
        sub_sphere_indices = torch.tensor([483, 402, 404, 406, 417, 410, 411, 413, 416, 421, 420, 412, 424, 426,
        428, 429, 432, 427, 436, 438, 440, 442, 444, 445, 448, 450, 452, 454,
        456, 458, 460, 409, 464, 466, 457, 470, 472, 474, 476, 485, 480, 482,
        484, 486, 488, 490, 492, 494, 451])

        # plt.figure(figsize=(10, 7))
        # ax = plt.axes(projection='3d')
        #
        # # Indices of nodes to color blue
        # blue_indices = sub_sphere_indices
        # print(blue_indices)
        #
        # # Create a color array
        # colors = ["red"] * len(sphere)  # Default color is red
        # for idx in blue_indices:
        #     colors[idx - len(mesh)] = "blue"  # Change color to blue for specified indices
        #
        # # ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], label="Mesh Points", color="blue")
        # ax.scatter(sphere[:, 0], sphere[:, 1], sphere[:, 2], c=colors, label="Collider Points")
        # # Label points
        # # for idx, (x, y, z) in enumerate(mesh):
        # #     ax.text(x, y, z, str(idx), fontsize=8, ha='right', va='bottom', color='red')
        # for idx, (x, y, z) in enumerate(sphere):
        #     ax.text(x, y, z, str(idx + len(mesh)), fontsize=8, ha='right', va='bottom', color='red')
        #
        # # plot faces
        # # for face in faces["deformable"]:
        # #     face = np.append(face, face[0])
        # #     ax.plot(mesh[face, 0], mesh[face, 1], mesh[face, 2], color="black")
        # for face in faces["collider"]:
        #     face = np.append(face, face[0]) - np.max(faces["deformable"]) - 1
        #     ax.plot(sphere[face, 0], sphere[face, 1], sphere[face, 2], color="black")
        #
        # # for edge in self.mesh_col_indices.T:
        # #     ax.plot([mesh[edge[0], 0], sphere[edge[1] - len(mesh), 0]], [mesh[edge[0], 1], sphere[edge[1] - len(mesh), 1]], [mesh[edge[0], 2], sphere[edge[1] - len(mesh), 2],], color="black")
        #
        # # Add legend and display
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        self.mesh_col_indices = torch.cartesian_prod(inner_mesh_indices, sub_sphere_indices).T
        self.col_mesh_indices = torch.cartesian_prod(sphere_indices, inner_mesh_indices).T
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
        (self.mesh_mesh_indices).shape[1] + (self.col_col_indices).shape[1] + (self.mesh_col_indices).shape[1]:,
        3] = 1.0
        self.base_edge_features = edge_features
        self.edge_id = torch.zeros(self.edge_indices.shape[1], dtype=torch.long)
        self.edge_id[:self.mesh_mesh_indices.shape[1]] = 0
        self.edge_id[
        self.mesh_mesh_indices.shape[1]:(self.mesh_mesh_indices).shape[1] + (self.col_col_indices).shape[1]] = 1
        self.edge_id[
        (self.mesh_mesh_indices).shape[1] + (self.col_col_indices).shape[1]:(self.mesh_mesh_indices).shape[1] +
                                                                            (self.col_col_indices).shape[1] +
                                                                            (self.mesh_col_indices).shape[1]] = 2
        self.edge_id[
        (self.mesh_mesh_indices).shape[1] + (self.col_col_indices).shape[1] + (self.mesh_col_indices).shape[1]:] = 3

    def normalization(self, x):
        return (x - self.min_x) / (self.max_x - self.min_x)

    def get(self, task_idx):
        task_dict = self.tasks[task_idx]
        # all pos
        mesh_pos = torch.stack(
            [task_dict["trajs"][traj_key]["cloth_positions"] for traj_key in sorted(task_dict["trajs"].keys())])
        collider_pos = torch.stack(
            [task_dict["trajs"][traj_key]["sphere_positions"] for traj_key in sorted(task_dict["trajs"].keys())])
        # normalization
        mesh_pos = self.normalization(mesh_pos)
        collider_pos = self.normalization(collider_pos)
        x = torch.cat((mesh_pos, collider_pos), dim=2)
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
            result["regression_features"] = regression_features
        return result

    def get_current_node_features(self, traj_dict, task_dict, x):
        # idea: set for all node index 0 to 1, since we want to control all of them.
        # 1: output node, 2: sub index cloth, 3:  sub index sphere
        h_description = ["one_hot", "one_hot", "cloth_identifier", "sphere_identifier", "boundary"]
        init_pos = traj_dict["cloth_positions"][0]
        # 3 node ids: deformable node: 0, collider_node: 1 output encoder node: 2
        node_id = torch.zeros(x.shape[2], 4)
        node_id[:, 0] = 1.0  # all are deformable
        # no output node
        node_id[0:len(init_pos), 2] = 1.0  # distinguish between cloth
        node_id[len(init_pos):, 3] = 1.0  # and sphere
        regression_features = None
        current_h = node_id
        boundary_features = torch.zeros(x.shape[2], 1)
        boundary_features[self.border_indices] = 1.0
        current_h = torch.cat([current_h, boundary_features], dim=-1)
        if self.config.material_properties or self.config.regression:
            # add material proprties
            sphere_density = torch.tensor([task_dict["params"]["sphere_density"]])
            sphere_density = sphere_density / 100.0  # max density is 100
            material_properties = torch.cat([sphere_density], dim=-1)
            if self.config.regression:
                regression_features = material_properties
            if self.config.material_properties:
                material_properties = material_properties.repeat(x.shape[2], 1)
                current_h = torch.cat([current_h, material_properties], dim=-1)
                h_description.append("sphere_density")
        return current_h, h_description, regression_features

    def get_faces(self):
        faces = self.global_data["cell_indices"]
        faces = {key: to_numpy(face) for key, face in faces.items()}
        faces["deformable"] = faces["cloth_indices"]
        faces["collider"] = faces["sphere_indices"] + np.max(faces["deformable"]) + 1
        del faces["sphere_indices"]
        del faces["cloth_indices"]
        return faces

    @property
    def traj_length(self):
        return 100
