import torch

from mango.algorithm.abstract_ml_algorithm import AbstractMLAlgorithm
from mango.util.own_types import ConfigDict


class MangoRegression(AbstractMLAlgorithm):
    def __init__(self, config: ConfigDict, train_dl, train_ds, eval_ds: torch.utils.data.Dataset):
        super().__init__(config, train_dl, train_ds, eval_ds)
        self.criterion = torch.nn.MSELoss()

        example_input_batch = train_ds[0]
        latent_dimension = config.simulator.encoder.latent_dimension
        if config.mat_prop_activation == "tanh":
            activation = torch.nn.Tanh()
        elif config.mat_prop_activation == "relu":
            activation = torch.nn.ReLU()
        elif config.mat_prop_activation == "leakyrelu":
            activation = torch.nn.LeakyReLU()
        else:
            raise ValueError(f"Activation function {config.mat_prop_activation} unknown.")
        self.mlp_out = torch.nn.Sequential(
            torch.nn.Linear(latent_dimension, latent_dimension),
            activation,  # use better activation function like leakyrelu?
            torch.nn.Linear(latent_dimension, example_input_batch["regression_features"].shape[-1])
        )

    def _training_step(self, batch, batch_idx):
        encoded_context_batch = self._encoder(batch)
        encoded_context_batch = encoded_context_batch[None] if len(encoded_context_batch.shape) == 1 else encoded_context_batch
        mat_out = self.mlp_out(encoded_context_batch)
        loss = self.criterion(mat_out, batch["regression_features"])
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            encoded_context_batch = self._encoder(batch)
            mat_out = self.mlp_out(encoded_context_batch[None])
            loss = self.criterion(mat_out, batch["regression_features"])
            if self.config.verbose:
                print(f"Task {batch_idx}, Prediction: {mat_out[0]}, Target: {batch['regression_features'][0]}")
        metric_results = {"mse": loss}
        result = {"metrics": metric_results, "visualizations": {}}
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
            self.log(f"val_{metric_name}", val_loss, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.clear()  # free memory

    def predict_trajectories(self, batch) -> torch.Tensor:
        return torch.empty(0)
