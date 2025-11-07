import torch

from mango.algorithm.abstract_ml_algorithm import AbstractMLAlgorithm
from mango.dataset.util.graph_input_output_util import unpack_ml_batch, get_deformable_mask, \
    get_deformable_pos
from mango.simulator import AbstractMLSimulator
from mango.util.own_types import ConfigDict


class NoMLMGNTorchGeometric(AbstractMLAlgorithm):
    def __init__(self, config: ConfigDict, simulator: AbstractMLSimulator, train_ds, eval_ds: torch.utils.data.Dataset):
        super().__init__(config, simulator, train_ds, eval_ds)
        self.criterion = torch.nn.MSELoss()

    def _training_step(self, batch, batch_idx):
        # encoder is dummy, only  use decoder
        vel_prediction = self._decoder(batch)
        def_mask = get_deformable_mask(batch)
        prediction = batch.pos[def_mask] + vel_prediction
        ground_truth = batch.y
        # only select target trajs
        loss = self.criterion(prediction, ground_truth)
        return loss

    def predict_trajectories(self, batch) -> torch.Tensor:
        # do roll out
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(batch, remove_batch_dim=True)
        h_description = batch["h_description"]
        h_description = [value[0] for value in h_description]
        output_trajectories = []
        for traj_idx in target_trajs:
            target_x = x[traj_idx, 0]  # initial pos
            target_v = v[traj_idx, 0]  # initial v
            target_h = h[traj_idx]
            eval_traj = self._train_ds.transform_to_torch_geometric(
                {
                    "x": target_x,  # shape (num_nodes, world_dim)
                    "v": target_v,  # shape (num_nodes, world_dim)
                    "h": target_h,  # shape (num_nodes, node_feature_dim)
                    "h_description": h_description,
                    "edge_indices": edge_indices,  # shape (2, num_edges)
                    "edge_features": edge_features,  # shape (num_edges, num_edge_features)
                    "y": torch.zeros_like(target_x),  # shape (num_nodes, world_dim) dummy target since it is not needed
                }
            )
            eval_traj.traj_pos = x[traj_idx]
            deformable_mask = get_deformable_mask(eval_traj)
            traj_def_pos = eval_traj.traj_pos[:, deformable_mask]
            initial_def_pos = get_deformable_pos(eval_traj)
            # get positions of the first step, save output
            output_trajectory = [torch.clone(initial_def_pos)]
            # predict the remaining steps in the future
            trajectory_length = traj_def_pos.shape[0]
            with torch.no_grad():
                for current_step in range(1, trajectory_length - 1):  # last step does not need update
                    predicted_dynamics = self._decoder(eval_traj)

                    predicted_def_pos = get_deformable_pos(eval_traj) + predicted_dynamics
                    # add updated mesh positions to output trajectories
                    output_trajectory.append(torch.clone(predicted_def_pos))
                    eval_traj = self._eval_ds.update(eval_traj, predicted_def_pos, current_step, edge_features)

            # final step
            predicted_dynamics = self._decoder(eval_traj)
            predicted_def_pos = get_deformable_pos(eval_traj) + predicted_dynamics
            # add updated mesh positions to output trajectories
            output_trajectory.append(torch.clone(predicted_def_pos))
            # finalize output trajectories
            output_trajectory = torch.stack(output_trajectory, dim=0)
            output_trajectories.append(output_trajectory)
        return torch.stack(output_trajectories, dim=0)
