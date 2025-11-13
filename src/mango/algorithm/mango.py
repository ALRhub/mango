
import torch

from mango.algorithm.abstract_ml_algorithm import AbstractMLAlgorithm
from mango.util.own_types import ConfigDict


class Mango(AbstractMLAlgorithm):
    def __init__(self, config: ConfigDict, train_dl, train_ds, eval_ds: torch.utils.data.Dataset):
        super().__init__(config, train_dl, train_ds, eval_ds)
        self.criterion = torch.nn.MSELoss()

    def _training_step(self, batch, batch_idx):
        encoded_context_batch = self._encoder(batch)
        if encoded_context_batch.isnan().any():
            print("NAN in encoder output")
        prediction = self._decoder(batch, encoded_context_batch)
        if prediction.isnan().any():
            print("NAN in decoder output")
        ground_truth = batch["x"][0]
        # only select target trajs
        ground_truth = ground_truth[batch["target_trajs"][0]]
        # only select deformable nodes
        deformable_mask = batch["h"][0, 0, :, 0] == 1
        ground_truth = ground_truth[:, :, deformable_mask, :]
        loss = self.criterion(prediction, ground_truth)
        return loss

    def predict_trajectories(self, batch) -> torch.Tensor:
        with torch.no_grad():
            # timed prediction
            # # Create events for timing
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            # # Record start time
            # start_event.record()
            # encoded_context_batch = self._encoder(batch)
            # # Record end time
            # end_event.record()
            # torch.cuda.synchronize()
            # # Compute elapsed time in milliseconds
            # elapsed_time = start_event.elapsed_time(end_event)
            # print(f"Time taken for encoding: {elapsed_time:.3f} ms")
            #
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            # # Record start time
            # start_event.record()
            # prediction = self._decoder(batch, encoded_context_batch)
            # end_event.record()
            # torch.cuda.synchronize()
            # # Compute elapsed time in milliseconds
            # elapsed_time = start_event.elapsed_time(end_event)
            # print(f"Time taken for decoding: {elapsed_time:.3f} ms")

            # standard prediction
            encoded_context_batch = self._encoder(batch)
            prediction = self._decoder(batch, encoded_context_batch)
        return prediction
