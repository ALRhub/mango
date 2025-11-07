import torch

from mango.dataset.util.graph_input_output_util import unpack_ml_batch
from mango.simulator.gnn.egno.egno import EGNO
from .abstract_encoder import AbstractEncoder

# This class is outdated and needs to be updated
class EGNOEncoder(AbstractEncoder):

    def __init__(self, config, example_input_batch):
        super().__init__(config)
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(
            example_input_batch, remove_batch_dim=True)
        self.egno_backbone = EGNO(
            n_layers=config.n_layers,
            in_node_nf=h.shape[-1],
            in_edge_nf=edge_features.shape[-1],
            hidden_nf=config.latent_dimension,
            use_time_conv=True,
            num_timesteps=x.shape[1],
            time_emb_dim=config.time_emb_dim
        )
        self.node_aggregation = config.node_aggregation


    def forward(self, batch) -> torch.Tensor:
        if self.node_aggregation == "virtual_node":
            x, v, h, h_description, edge_indices, edge_features, context_trajs = self.add_encoder_output_node(batch)
        else:
            x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(batch,
                                                                                                remove_batch_dim=True)
        # iterator over context trajs indices and compute the context encoding.
        context_outputs = []
        for context_idx in context_trajs:
            context_x = x[context_idx]
            context_v = v[context_idx]
            context_h = h[context_idx]
            num_timesteps = context_x.shape[0]
            num_nodes = context_x.shape[1]
            world_dim = context_x.shape[-1]
            _, _, all_nodes_output = self.egno_backbone(context_x, context_h, edge_indices, edge_features, v=context_v)
            all_nodes_output = all_nodes_output.view(num_timesteps, num_nodes, self.config.latent_dimension)
            # take only the context output node at the last time step
            if self.node_aggregation == "virtual_node":
                context_output = all_nodes_output[-1, -1, :]
            elif self.node_aggregation == "mean":
                all_nodes_output = all_nodes_output[-1, :, :]
                context_output = torch.mean(all_nodes_output, dim=0)
            elif self.node_aggregation == "max":
                all_nodes_output = all_nodes_output[-1, :, :]
                context_output = torch.max(all_nodes_output, dim=0).values
            else:
                raise ValueError(f"Node aggregation method {self.node_aggregation} not supported")
            context_outputs.append(context_output)
        # max aggregation
        context_outputs = torch.stack(context_outputs, dim=0)
        context_output = torch.max(context_outputs, dim=0).values
        return context_output
