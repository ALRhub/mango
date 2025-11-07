import torch
from torch_geometric.data import Batch

from mango.simulator.ml_encoder.abstract_encoder import AbstractEncoder


class DummyEncoder(AbstractEncoder):
    """
    Dummy Encoder which just returns 0 to compare how good Meta Learning is in comparison.
    """
    def __init__(self, config, example_input_batch):
        super().__init__(config)

    def forward(self, batch) -> torch.Tensor:
        if isinstance(batch, dict):
            x = batch["x"]
        elif isinstance(batch, Batch):
            x = batch.x
        else:
            raise ValueError(f"Unknown batch type: {type(batch)}")
        return torch.zeros(self.config.latent_dimension).to(x)


