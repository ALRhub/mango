import torch
from mango.algorithm.ltsgns_v2 import LTSGNSV2
from mango.util.own_types import ConfigDict


class NoMLMGN(LTSGNSV2):
    def __init__(self, config: ConfigDict, train_dl, train_ds, eval_ds: torch.utils.data.Dataset):
        super().__init__(config, train_dl, train_ds, eval_ds)
        self.criterion = torch.nn.MSELoss()

    def _training_step(self, batch, batch_idx):
        encoded_context_batch = self._encoder(batch)
        assert encoded_context_batch.allclose(torch.zeros_like(encoded_context_batch)), "Encoding should be 0"
        prediction = self._decoder(batch, encoded_context_batch)
        if prediction.isnan().any():
            print("NAN in decoder output")
        # todo: correct ground truth
        ground_truth = batch["y"]
        loss = self.criterion(prediction, ground_truth)
        return loss

