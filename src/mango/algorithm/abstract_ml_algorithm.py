import os
from abc import ABC, abstractmethod
from typing import Any

import hydra
import torch
from lightning import LightningModule

from mango.algorithm.util.get_optimizer import _get_optimizer, _get_scheduler
from mango.algorithm.util.util import get_plotly_figure_from_step_losses
from mango.dataset.util.graph_input_output_util import unpack_ml_batch
from mango.logger.visualizations.log_visualizations import visualize_ml_trajectories
from mango.simulator import get_simulator
from mango.util.own_types import ConfigDict, ValueDict


class AbstractMLAlgorithm(LightningModule, ABC):
    """
    Abstract class for the full algorithm, including the Simulator.
    """

    def __init__(self, config: ConfigDict,  train_dl, train_ds: torch.utils.data.Dataset, eval_ds: torch.utils.data.Dataset):
        super().__init__()
        self._config: ConfigDict = config
        simulator = get_simulator(config.simulator, train_dl, eval_ds)
        simulator = torch.compile(simulator)
        self._encoder = simulator.encoder
        self._decoder = simulator.decoder
        self._simulator = simulator
        self._train_ds = train_ds
        self._eval_ds = eval_ds
        self._vis_path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "visualizations")

        # temp dicts to save eval results
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.best_val_mse = 1000000.0

        # save config as hyperparameter
        self.save_hyperparameters("config")

    def training_step(self, batch, batch_idx):
        loss = self._training_step(batch, batch_idx)
        self.training_step_outputs.append(loss)
        return loss

    @abstractmethod
    def _training_step(self, batch, batch_idx) -> torch.Tensor:
        raise NotImplementedError

    def on_train_epoch_start(self):
        print("Training Epoch: ", self.current_epoch)

    def on_train_epoch_end(self):
        # `outputs` is a list of losses from the `training_step` for each batch
        # Calculate the mean loss for the epoch
        epoch_average = torch.stack(self.training_step_outputs).mean()
        # Log the mean epoch loss to WandB
        self.log('train_loss', epoch_average, on_epoch=True, prog_bar=True)
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        visualize = batch_idx in self.config.evaluator.animation_indices
        predicted_trajectories = self.predict_trajectories(batch)
        metric_results = self.evaluate_trajectories(predicted_trajectories, batch)
        vis_results = {}
        if visualize:
            # take the last target index as example. This is usually not in the context set.
            predicted_trajectory = predicted_trajectories[-1]
            vis_results[f"Traj_{batch_idx}"] = {"eval_traj": {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in batch.items()},
                                                "predicted_traj": predicted_trajectory.cpu(),
                                                "predicted_traj_index": batch["target_trajs"][0, -1].item()}
        result = {"metrics": metric_results, "visualizations": vis_results}
        self.validation_step_outputs.append(result)
        return result

    def evaluate_trajectories(self, predicted_trajectory: torch.Tensor, batch: dict) -> ValueDict:
        """
        Evaluates the predicted trajectories against the ground truth trajectories.
        Returns: The evaluation results as a ValueDict.
        """
        # get the correct evaluation interval of ground truth and predicted trajectory + unsqueeze

        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(batch, remove_batch_dim=True)
        gth_pos = x[target_trajs]
        deformable_mask = h[0, :, 0] == 1
        gth_pos = gth_pos[:, :, deformable_mask, :]

        pred_pos = predicted_trajectory

        assert gth_pos.shape == pred_pos.shape, "Shapes of ground truth and predicted trajectory do not match."
        result_dict = {}
        for metric_idx, metric in enumerate(self.config.evaluator.metric):
            individual_result = self._eval_single_metric(pred_pos,
                                                         gth_pos,
                                                         metric)
            result_dict[metric] = individual_result
        return result_dict

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        # metrics
        metrics = [output["metrics"] for output in outputs]
        current_val_mse = None
        for metric_name in metrics[0].keys():
            metric_values = [metric[metric_name] for metric in metrics]
            metric_values = torch.stack(metric_values)  # shape (num_batches, len(traj))
            val_loss = metric_values.mean()
            if metric_name == "mse":
                current_val_mse = val_loss
            per_step_loss = metric_values.mean(dim=0)
            plotly_fig = get_plotly_figure_from_step_losses(per_step_loss, f"{metric_name} Loss over trajectory steps")
            if self.logger is not None:
                self.logger.experiment.log({f"val_{metric_name}_plot": plotly_fig})
                self.log(f"val_{metric_name}", val_loss, on_epoch=True, prog_bar=True)

        if current_val_mse < self.best_val_mse:
            self.best_val_mse = current_val_mse
            # visualizations: only log if the current method has the lower val_mse of all time
            visualizations = [output["visualizations"] for output in outputs]
            # flatten the list of dicts
            visualizations = {k: v for vis_dict in visualizations for k, v in vis_dict.items()}
            visualize_ml_trajectories(visualizations, self.current_epoch, self._vis_path, self._eval_ds)
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        visualize = batch_idx in self.config.evaluator.animation_indices
        target_trajs = torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]])
        batch_output = []
        vis_results = {}
        for context_size in range(1, 9):
            context_trajs = torch.tensor([[i for i in range(context_size)]])
            batch["context_trajs"] = context_trajs
            batch["target_trajs"] = target_trajs
            predicted_trajectories = self.predict_trajectories(batch)
            metric_results = self.evaluate_trajectories(predicted_trajectories, batch)
            batch_output.append(metric_results["mse"])
            if visualize:
                vis_results[f"context_size_{context_size}"] = {}
                # take the last target index as example. This is usually not in the context set.
                predicted_trajectory = predicted_trajectories[-1]
                vis_results[f"context_size_{context_size}"][f"Traj_{batch_idx}"] = {
                    "eval_traj": {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in
                                  batch.items()},
                    "predicted_traj": predicted_trajectory.cpu(),
                    "predicted_traj_index": batch["target_trajs"][0, -1].item(),
                    "context_traj_index": batch["context_trajs"][0].cpu() if context_size == 8 else None}

        result = {"test_loss": torch.stack(batch_output), "visualizations": vis_results}
        self.test_step_outputs.append(result)

    def on_test_end(self) -> None:
        outputs = self.test_step_outputs
        visualizations = [output["visualizations"] for output in outputs]
        # average over all test batches
        outputs = torch.stack([output["test_loss"] for output in outputs])
        metrics = outputs.mean(dim=0)
        output = {
            "metrics": metrics,
            "config": self.config,
            "visualizations": visualizations,
        }
        self.logger.log_metrics(output, self.current_epoch)
        self.test_step_outputs.clear()  # free memory



    def configure_optimizers(self):
        optimizer = _get_optimizer(config=self.config.optimizer, simulator=self.simulator)
        scheduler = _get_scheduler(config=self.config.scheduler, optimizer=optimizer)
        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": self.config.check_val_every_n_epoch,
            "monitor": "val_mse",
            "strict": False,
        }
        return [optimizer], [scheduler]

    @abstractmethod
    def predict_trajectories(self, batch) -> torch.Tensor:
        raise NotImplementedError


    def _eval_single_metric(self, predicted_mesh_positions, gth_mesh_positions, metric) -> torch.Tensor:
        """
        Evaluates a single metric for a single time interval.

        Returns: The evaluation result as a Float.

        """
        if metric == "mse":
            # average over different trajectories, nodes and world dim, not over time!
            mse = torch.mean((predicted_mesh_positions - gth_mesh_positions) ** 2, dim=[0, -1, -2])  # shape (len(traj),)
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
