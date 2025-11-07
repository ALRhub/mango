import torch

from mango.dataset.util.graph_input_output_util import unpack_ml_batch
from torch_geometric.nn import MLP
import torch.nn as nn
import math

from .abstract_encoder import AbstractEncoder

class TransformerEncoder(AbstractEncoder):
    def __init__(self, config, example_input_batch):
        super().__init__(config)
        self.input_dim = example_input_batch["x"].shape[-1] + example_input_batch["v"].shape[-1] + example_input_batch["h"].shape[-1]
        self.embedding = MLP([self.input_dim, config.latent_dimension], norm=None)

        spatial_cfg = config.spatial_module
        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, config.latent_dimension), requires_grad=True)
        self.spatial_embedding = nn.Linear(config.latent_dimension, config.latent_dimension)
        self.spatial_transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.latent_dimension,
            nhead=spatial_cfg.transformer_n_heads,
            dim_feedforward=spatial_cfg.transformer_hidden_size,
            dropout=spatial_cfg.transformer_dropout,
        )
        self.spatial_transformer_encoder = nn.TransformerEncoder(self.spatial_transformer_encoder_layer, num_layers=spatial_cfg.transformer_n_layers)
            
        self.temporal_module = config.temporal_module
        if self.temporal_module == "transformer":
            temporal_cfg = config.temporal_module_transformer
            self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, config.latent_dimension), requires_grad=True)
            self.temporal_embedding = nn.Linear(config.latent_dimension, config.latent_dimension)
            self.temporal_transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.latent_dimension,
                nhead=temporal_cfg.transformer_n_heads,
                dim_feedforward=temporal_cfg.transformer_hidden_size,
                dropout=temporal_cfg.transformer_dropout,
            )
            self.temporal_transformer_encoder = nn.TransformerEncoder(self.temporal_transformer_encoder_layer, num_layers=temporal_cfg.transformer_n_layers)
            if temporal_cfg.transformer_pos_embedding:
                self.positional_embedding = self.create_positional_embedding(config.latent_dimension, example_input_batch["x"].shape[2])
            else:
                self.positional_embedding = 0.0
        elif self.temporal_module == "deepset":
            self.time_mlp_inner = MLP([config.latent_dimension, config.latent_dimension], norm="layer_norm")
            self.time_mlp_outer = MLP([config.latent_dimension, config.latent_dimension], norm="layer_norm")

    def create_positional_embedding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, batch) -> torch.Tensor:
        x, v, h, h_description, edge_indices, edge_features, context_trajs, _ = unpack_ml_batch(batch, remove_batch_dim=True)

        # iterator over context trajs indices and compute the context encoding.
        context_outputs = []
        for context_idx in context_trajs:
            context_x = x[context_idx]
            context_v = v[context_idx]
            context_h = h[context_idx]
            num_timesteps = context_x.shape[0]

            # Transformer Encoder for spatial information
            input_vector = torch.cat([context_x, context_v, context_h.repeat(num_timesteps, 1, 1)], dim=-1)
            all_nodes_output = self.embedding(input_vector)

            all_nodes_output = self.spatial_embedding(all_nodes_output)
            spatial_cls_token = self.spatial_cls_token.expand(num_timesteps, -1, -1).to(all_nodes_output.device)
            all_nodes_output = torch.cat([spatial_cls_token, all_nodes_output], dim=1)
            all_nodes_output = all_nodes_output.permute(1, 0, 2)
            all_nodes_output = self.spatial_transformer_encoder(all_nodes_output)
            all_nodes_output = all_nodes_output[0, :]
            
            if self.temporal_module == "transformer":
                # Transformer Encoder for temporal information
                all_nodes_output = all_nodes_output[None]
                all_nodes_output = self.temporal_embedding(all_nodes_output)
                all_nodes_output = all_nodes_output + self.positional_embedding
                all_nodes_output = torch.cat([self.temporal_cls_token.to(all_nodes_output.device), all_nodes_output], dim=1)
                all_nodes_output = all_nodes_output.permute(1, 0, 2)
                all_nodes_output = self.temporal_transformer_encoder(all_nodes_output)
                context_output = all_nodes_output[0, 0, :]  # take the cls token output
            elif self.temporal_module == "deepset":
                # DeepSet aggregation of all time steps
                all_nodes_output = self.time_mlp_inner(all_nodes_output)
                all_nodes_output = all_nodes_output.sum(dim=0)
                context_output = self.time_mlp_outer(all_nodes_output)
            else:
                context_output = all_nodes_output[-1, :]  # take the last time step output
            
            context_outputs.append(context_output)
        # max aggregation
        context_outputs = torch.stack(context_outputs, dim=0)
        context_output = torch.max(context_outputs, dim=0).values
        return context_output
