import torch


class AbstractMLSimulator(torch.nn.Module):
    """
    Container class for the Meta Learning simulator. No functionality in this class, only a way to reference to the
    config and encoder and decoder. Also useful to provide the same interface for all simulators.
    """
    def __init__(self, config, encoder, decoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
