import os
from abc import ABC, abstractmethod

import hydra
import torch
from torch_geometric.data import Batch
from lightning import LightningModule

from mango.algorithm.util.get_optimizer import _get_optimizer
from mango.algorithm.util.util import get_plotly_figure_from_step_losses
from mango.dataset.util.graph_input_output_util import get_deformable_mask
from mango.logger.visualizations.log_visualizations import visualize_trajectories
from mango.util.own_types import ConfigDict, ValueDict


class AbstractAlgorithm(LightningModule, ABC):
    """
    Abstract class for the full algorithm, including the Simulator.
    """

    def __init__(self, config: ConfigDict, simulator: torch.nn.Module, train_ds, eval_ds: torch.utils.data.Dataset):
        super().__init__()
        self._config: ConfigDict = config
        self._simulator = simulator
        self._train_ds = train_ds
        self._eval_ds = eval_ds
        self._vis_path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "visualizations")

        # temp dicts to save eval results
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        loss = self._training_step(batch, batch_idx)
        self.training_step_outputs.append(loss)
        return loss

    @abstractmethod
    def _training_step(self, batch, batch_idx) -> torch.Tensor:
        raise NotImplementedError

    def on_train_epoch_end(self):
        # `outputs` is a list of losses from the `training_step` for each batch
        # Calculate the mean loss for the epoch
        epoch_average = torch.stack(self.training_step_outputs).mean()
        # Log the mean epoch loss to WandB
        self.log('train_loss', epoch_average, on_epoch=True, prog_bar=True)
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        visualize = batch_idx in self.config.evaluator.animation_indices
        predicted_trajectory = self.predict_trajectory(batch)
        metric_results = self.evaluate_trajectory(predicted_trajectory, batch)
        vis_results = {}
        if visualize:
            # save it in the eval_dict
            vis_results[f"Traj_{batch_idx}"] = {"eval_traj": batch.cpu(),
                                                "predicted_traj": predicted_trajectory.cpu()}
        result = {"metrics": metric_results, "visualizations": vis_results}
        self.validation_step_outputs.append(result)
        return result

    def on_validation_epoch_end(self):

        outputs = self.validation_step_outputs
        # metrics
        metrics = [output["metrics"] for output in outputs]
        for metric_name in metrics[0].keys():
            metric_values = [metric[metric_name] for metric in metrics]
            metric_values = torch.stack(metric_values)  # shape (num_trajs, len(traj))
            val_loss = metric_values.mean()
            per_step_loss = metric_values.mean(dim=0)
            plotly_fig = get_plotly_figure_from_step_losses(per_step_loss, f"{metric_name} Loss over trajectory steps")
            self.logger.experiment.log({f"val_{metric_name}_plot": plotly_fig})
            self.log(f"val_{metric_name}", val_loss, on_epoch=True, prog_bar=True)

        # visualizations
        visualizations = [output["visualizations"] for output in outputs]
        # flatten the list of dicts
        visualizations = {k: v for vis_dict in visualizations for k, v in vis_dict.items()}
        visualize_trajectories(visualizations, self.current_epoch, self._vis_path, self._eval_ds)

        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = _get_optimizer(config=self.config.optimizer, simulator=self.simulator)
        return optimizer

    @abstractmethod
    def predict_trajectory(self, traj: Batch) -> torch.Tensor:
        raise NotImplementedError

    def evaluate_trajectory(self, predicted_trajectory: torch.Tensor, eval_traj: Batch) -> ValueDict:
        """
        Evaluates the predicted trajectory against the ground truth trajectory.
        Returns: The evaluation results as a ValueDict.

        """
        # get the correct evaluation interval of ground truth and predicted trajectory + unsqueeze
        deformable_mask = get_deformable_mask(eval_traj)
        gth_def_position = eval_traj.traj_pos[:, deformable_mask, :]
        predicted_def_positions = predicted_trajectory
        assert gth_def_position.shape == predicted_def_positions.shape, "Shapes of ground truth and predicted trajectory do not match."
        result_dict = {}
        for metric_idx, metric in enumerate(self.config.evaluator.metric):
            individual_result = self._eval_single_metric(predicted_def_positions,
                                                         gth_def_position,
                                                         metric)
            result_dict[metric] = individual_result
        return result_dict

    def _eval_single_metric(self, predicted_mesh_positions, gth_mesh_positions, metric) -> torch.Tensor:
        """
        Evaluates a single metric for a single time interval.

        Returns: The evaluation result as a Float.

        """
        if metric == "mse":
            mse = torch.mean((predicted_mesh_positions - gth_mesh_positions) ** 2, dim=[1, 2])  # shape (len(traj),)
            return mse
        else:
            raise ValueError("Unknown metric: {}".format(metric))

    @property
    def simulator(self) -> torch.nn.Module:
        if self._simulator is None:
            raise ValueError("Simulator not set")
        return self._simulator

    @property
    def config(self) -> ConfigDict:
        return self._config
