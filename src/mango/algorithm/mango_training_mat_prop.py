import torch

from mango.algorithm.mango import Mango
from mango.util.own_types import ConfigDict


class MangoTrainingMatProp(Mango):
    def __init__(self, config: ConfigDict, train_dl, train_ds, eval_ds: torch.utils.data.Dataset):
        super().__init__(config, train_dl, train_ds, eval_ds)
        self.criterion = torch.nn.MSELoss()

        example_input_batch = train_ds[0]
        latent_dimension = self.config.simulator.encoder.latent_dimension
        if self.config.mat_prop_activation == "tanh":
            activation = torch.nn.Tanh()
        elif self.config.mat_prop_activation == "relu":
            activation = torch.nn.ReLU()
        elif self.config.mat_prop_activation == "leakyrelu":
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
        if encoded_context_batch.isnan().any():
            print("NAN in encoder output")
        prediction = self._decoder(batch, encoded_context_batch)
        if prediction.isnan().any():
            print("NAN in decoder output")

        encoded_context_batch = encoded_context_batch[None] if len(
            encoded_context_batch.shape) == 1 else encoded_context_batch
        mat_out = self.mlp_out(encoded_context_batch)
        mat_loss = self.criterion(mat_out, batch["regression_features"])
        ground_truth = batch["x"][0]
        # only select target trajs
        ground_truth = ground_truth[batch["target_trajs"][0]]
        # only select deformable nodes
        deformable_mask = batch["h"][0, 0, :, 0] == 1
        ground_truth = ground_truth[:, :, deformable_mask, :]
        ml_loss = self.criterion(prediction, ground_truth)
        return mat_loss + ml_loss
