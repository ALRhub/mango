import torch
from omegaconf import OmegaConf

from mango.simulator.decoder.decoder import Decoder
from mango.simulator.decoder.scaled_tanh import ScaledTanh
from mango.util.own_types import ConfigDict


def get_decoder(config: ConfigDict, action_dim: int, input_dimension: int,
                simulator_class: str) -> torch.nn.Module:
    """
    Build a decoder, which consists of a decode_module and a readout_module.
    Args:
        config:
        action_dim:
        device:
        input_dimensions: Dictionary containing the dimensions of the input features
        simulator_class:

    Returns: initialized Decoder

    """
    if simulator_class == "HMPNStepSimulator":
        from mango.simulator.util.mlp import MLP
        decode_module = MLP(in_features=input_dimension,
                            latent_dimension=config.latent_dimension,
                            config=OmegaConf.create(dict(activation_function="relu",
                                                         add_output_layer=False,
                                                         num_layers=1,
                                                         regularization={
                                                             "dropout": config.regularization.dropout,
                                                         },
                                                         )),
                            )
    else:
        raise NotImplementedError(f"Decoder for simulator class {simulator_class} not implemented")
    readout_module = torch.nn.Linear(config.latent_dimension, action_dim)
    if config.tanh.enabled:
        output_activation = ScaledTanh(config.tanh.scale_factor, config.tanh.input_scale_factor)
    else:
        output_activation = torch.nn.Identity()
    decoder = Decoder(decode_module, readout_module, output_activation)
    return decoder
