import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from mango.algorithm.abstract_algorithm import AbstractAlgorithm
from mango.dataset.util.graph_input_output_util import get_deformable_pos, get_deformable_mask
from mango.util.own_types import  ConfigDict


class MGN(AbstractAlgorithm):
    def __init__(self, config: ConfigDict, simulator: torch.nn.Module, train_ds, eval_ds: torch.utils.data.Dataset):
        super().__init__(config, simulator, train_ds, eval_ds)
        self.criterion = torch.nn.MSELoss()

    def _training_step(self, batch, batch_idx):
        # is usually velocities, or accelerations if second order dynamics are used
        predicted_dynamics = self.simulator(batch)
        deformable_pos = get_deformable_pos(batch)
        predicted_pos = deformable_pos + predicted_dynamics
        gth_pos = batch.y[:, :3]  # only predict position for now
        gth_pos = gth_pos[get_deformable_mask(batch)]

        loss = self.criterion(predicted_pos, gth_pos)
        return loss

    def predict_trajectory(self, eval_traj: Batch) -> torch.Tensor:
        assert len(eval_traj) == 1, "Testing only batch size 1 supported"
        eval_traj = eval_traj[0]

        deformable_mask = get_deformable_mask(eval_traj)
        traj_def_pos = eval_traj.traj_pos[:, deformable_mask]
        initial_def_pos = get_deformable_pos(eval_traj)

        # get positions of the first step, save output
        output_trajectories = [torch.clone(initial_def_pos)]

        # predict the remaining steps in the future
        trajectory_length = traj_def_pos.shape[0]

        with torch.no_grad():
            for current_step in tqdm(range(1, trajectory_length - 1), desc="Predicting Eval Trajectory..",
                                     disable=not self.config.verbose):  # last step does not need update
                predicted_dynamics = self.simulator(eval_traj)

                predicted_def_pos = get_deformable_pos(eval_traj) + predicted_dynamics
                # add updated mesh positions to output trajectories
                output_trajectories.append(torch.clone(predicted_def_pos))
                eval_traj = self._eval_ds.update(eval_traj, predicted_def_pos, current_step)

        # final step
        predicted_dynamics = self.simulator(eval_traj)
        predicted_def_pos = get_deformable_pos(eval_traj) + predicted_dynamics
        # add updated mesh positions to output trajectories
        output_trajectories.append(torch.clone(predicted_def_pos))

        # finalize output trajectories
        output_trajectories = torch.stack(output_trajectories, dim=0)
        return output_trajectories
