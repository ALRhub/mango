import torch

from mango.dataset.util.graph_input_output_util import unpack_ml_batch
from torch_geometric.nn import MLP

from .abstract_encoder import AbstractEncoder


class DeepSetEncoder(AbstractEncoder):

    def __init__(self, config, example_input_batch):
        super().__init__(config)

        self.input_dim = example_input_batch["x"].shape[-1] + example_input_batch["v"].shape[-1] + example_input_batch["h"].shape[-1]
        self.embedding = MLP([self.input_dim, config.latent_dimension], norm=None)
        self.node_mlp_inner = MLP([config.latent_dimension, config.latent_dimension, config.latent_dimension], norm="layer_norm")
        self.node_mlp_outer = MLP([config.latent_dimension, config.latent_dimension, config.latent_dimension], norm="layer_norm")

        self.time_mlp_inner = MLP([config.latent_dimension, config.latent_dimension], norm="layer_norm")
        self.time_mlp_outer = MLP([config.latent_dimension, config.latent_dimension], norm="layer_norm")

    def forward(self, batch) -> torch.Tensor:
        x, v, h, h_description, edge_indices, edge_features, context_trajs, _ = unpack_ml_batch(batch, remove_batch_dim=True)
        # iterator over context trajs indices and compute the context encoding.
        context_outputs = []
        time_subsampling = self.config.time_subsampling
        subsampled_x = x[:, ::time_subsampling]
        subsampled_v = v[:, ::time_subsampling]
        x = torch.cat([subsampled_x, x[:, -1:]], dim=1)
        v = torch.cat([subsampled_v, v[:, -1:]], dim=1)
        for context_idx in context_trajs:
            context_x = x[context_idx]
            context_v = v[context_idx]
            context_h = h[context_idx]
            num_timesteps = context_x.shape[0]
            num_nodes = context_x.shape[1]
            input_vector = torch.cat([context_x, context_v, context_h.repeat(num_timesteps, 1, 1)], dim=-1)
            all_nodes_output = self.embedding(input_vector)

            # DeepSet aggregation of all nodes
            all_nodes_output = self.node_mlp_inner(all_nodes_output)
            all_nodes_output = all_nodes_output.mean(dim=1)
            all_nodes_output = self.node_mlp_outer(all_nodes_output)

            # DeepSet aggregation of all time steps
            all_nodes_output = self.time_mlp_inner(all_nodes_output)
            all_nodes_output = all_nodes_output.mean(dim=0)
            context_output = self.time_mlp_outer(all_nodes_output)
            
            context_outputs.append(context_output)
        # max aggregation
        context_outputs = torch.stack(context_outputs, dim=0)
        context_output = torch.max(context_outputs, dim=0).values
        return context_output
