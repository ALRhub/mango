import torch

from mango.dataset.util.graph_input_output_util import unpack_ml_batch
from mango.simulator.mango import Mango
from mango.simulator.ml_encoder.abstract_encoder import AbstractEncoder


class MGNEncoder(AbstractEncoder):
    """
    Takes only the last step and does MGN on that to get a context encoding using mean or virtual node aggregation
    """
    def __init__(self, config, example_input_batch):
        super().__init__(config)
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(
            example_input_batch, remove_batch_dim=True)
        self.mgno_backbone = Mango(
            n_layers=config.n_layers,
            h_dim=h.shape[-1],
            edge_feature_dim=edge_features.shape[-1],
            world_dim=x.shape[-1],
            latent_dim=config.latent_dimension,
            activation=config.activation,
            scatter_reduce=config.scatter_reduce,
            use_hidden_layers=config.use_hidden_layers,
            use_time_conv=False,
        )
        self.node_aggregation = config.node_aggregation

    def forward(self, batch) -> torch.Tensor:
        if self.node_aggregation == "virtual_node":
            x, v, h, h_description, edge_indices, edge_features, context_trajs = self.add_encoder_output_node(batch)
        else:
            x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(batch,
                                                                                                               remove_batch_dim=True)
        # only use last timestep
        context_x = x[context_trajs, -1:, :, :]
        context_v = v[context_trajs, -1:, :, :]
        context_h = h[context_trajs]
        edge_features = edge_features[context_trajs]
        all_nodes_output = self.mgno_backbone(context_x, context_h, context_v, edge_indices, edge_features)
        if self.node_aggregation == "virtual_node":
            context_output = all_nodes_output[:, -1, -1, :]
        elif self.node_aggregation == "mean":
            all_nodes_output = all_nodes_output[:, -1, :, :]
            context_output = torch.mean(all_nodes_output, dim=1)
        elif self.node_aggregation == "max":
            all_nodes_output = all_nodes_output[:, -1, :, :]
            context_output = torch.max(all_nodes_output, dim=1).values
        else:
            raise ValueError(f"Node aggregation method {self.node_aggregation} not supported")
        context_output = torch.max(context_output, dim=0).values
        return context_output

