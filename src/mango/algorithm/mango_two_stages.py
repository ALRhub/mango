import torch

from mango.algorithm.mango_training_mat_prop import MangoTrainingMatProp
from mango.util.own_types import ConfigDict


class MangoTwoStages(MangoTrainingMatProp):
    def __init__(self, config: ConfigDict, train_dl, train_ds, eval_ds: torch.utils.data.Dataset):
        super().__init__(config, train_dl, train_ds, eval_ds)
        self.stages_switch_epoch = config.stages_switch_epoch
        self.frozen_encoder = False

    def _training_step(self, batch, batch_idx):
        if self.current_epoch < self.stages_switch_epoch:
            # first stage: only train mat prop encoder
            encoded_context_batch = self._encoder(batch)
            encoded_context_batch = encoded_context_batch[None] if len(
                encoded_context_batch.shape) == 1 else encoded_context_batch
            mat_out = self.mlp_out(encoded_context_batch)
            loss = self.criterion(mat_out, batch["regression_features"])
            return loss
        else:
            if not self.frozen_encoder:
                # freeze encoder
                for param in self._encoder.parameters():
                    param.requires_grad = False
                # also freeze mlp_out
                for param in self.mlp_out.parameters():
                    param.requires_grad = False
                print("Switching to stage 2: Freezing encoder and training decoder only.")
                self.frozen_encoder = True
            # only train decoder
            encoded_context_batch = self._encoder(batch)
            if encoded_context_batch.isnan().any():
                print("NAN in encoder output")
            mat_out = self.get_mat_out(encoded_context_batch)
            prediction = self._decoder(batch, mat_out)
            if prediction.isnan().any():
                print("NAN in decoder output")
            ground_truth = batch["x"][0]
            # only select target trajs
            ground_truth = ground_truth[batch["target_trajs"][0]]
            # only select deformable nodes
            deformable_mask = batch["h"][0, 0, :, 0] == 1
            ground_truth = ground_truth[:, :, deformable_mask, :]
            ml_loss = self.criterion(prediction, ground_truth)
            return ml_loss

    def predict_trajectories(self, batch) -> torch.Tensor:
        with torch.no_grad():
            # standard prediction
            encoded_context_batch = self._encoder(batch)
            mat_out = self.get_mat_out(encoded_context_batch)
            prediction = self._decoder(batch, mat_out)
        return prediction

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.current_epoch < self.stages_switch_epoch:
                encoded_context_batch = self._encoder(batch)
                mat_out = self.get_mat_out(encoded_context_batch)
                # add batch dim again
                mat_out = mat_out[None] if len(mat_out.shape) == 1 else mat_out
                loss = self.criterion(mat_out, batch["regression_features"])
                if self.config.verbose:
                    print(f"Task {batch_idx}, Prediction: {mat_out[0]}, Target: {batch['regression_features'][0]}")
                metric_results = {"mse_material": loss, "mse": torch.tensor(1000.0)}  # dummy value for mse
                result = {"metrics": metric_results, "visualizations": {}}
                self.validation_step_outputs.append(result)
            else:
                result = super().validation_step(batch, batch_idx)
        return result

    def on_validation_epoch_end(self):
        if self.current_epoch < self.stages_switch_epoch:
            print(f"End of epoch {self.current_epoch}. Stage 1: Evaluating material property prediction.")
            outputs = self.validation_step_outputs
            # metrics
            metrics = [output["metrics"] for output in outputs]
            for metric_name in metrics[0].keys():
                metric_values = [metric[metric_name] for metric in metrics]
                metric_values = torch.stack(metric_values)  # shape (num_trajs, len(traj))
                val_loss = metric_values.mean()
                self.log(f"val_{metric_name}", val_loss, on_epoch=True, prog_bar=True)

            self.validation_step_outputs.clear()  # free memory
        else:
            print(f"End of epoch {self.current_epoch}. Stage 2: Evaluating trajectory prediction.")
            super().on_validation_epoch_end()  # call parent method

    def get_mat_out(self, encoded_context_batch):
        encoded_context_batch = encoded_context_batch[None] if len(
            encoded_context_batch.shape) == 1 else encoded_context_batch
        mat_out = self.mlp_out(encoded_context_batch)
        # remove batch dim again
        mat_out = mat_out[0]
        return mat_out
