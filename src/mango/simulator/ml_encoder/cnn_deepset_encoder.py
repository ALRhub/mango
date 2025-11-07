import torch

from mango.dataset.util.graph_input_output_util import unpack_ml_batch
from torch_geometric.nn import MLP

from mango.simulator.ml_encoder.abstract_encoder import AbstractEncoder


class CNNDeepSetEncoder(AbstractEncoder):
    """
    Makes the time aggregation first and then does MGN on initial pos with computed time h
    """

    def __init__(self, config, example_input_batch):
        super().__init__(config)
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(
            example_input_batch, remove_batch_dim=True)
        self.centralize_input = config.get("centralize_input", False)
        self.z_pos_feature = config.get("z_pos_feature", False)
        self.input_dim = x.shape[-1] + v.shape[-1] + h.shape[-1] + 1 * self.z_pos_feature

        cnn_layers = [
            torch.nn.Conv1d(in_channels=self.input_dim, out_channels=config.latent_dimension,
                            kernel_size=3,
                            padding=1),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels=config.latent_dimension, out_channels=config.latent_dimension, kernel_size=3,
                            padding=1),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels=config.latent_dimension, out_channels=config.latent_dimension, kernel_size=3,
                            padding=1),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels=config.latent_dimension, out_channels=config.latent_dimension, kernel_size=3,
                            padding=1),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels=config.latent_dimension, out_channels=h.shape[-1], kernel_size=3,
                            padding=0),
            torch.nn.LeakyReLU(),
        ]
        if config.dataset_name == "torus_ml" and x.shape[1] > 60:
            # add more layers for torus
            cnn_layers = cnn_layers[:-2] + [
                torch.nn.Conv1d(in_channels=config.latent_dimension, out_channels=config.latent_dimension, kernel_size=3,
                               padding=1),
                torch.nn.MaxPool1d(kernel_size=2, stride=2),
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(in_channels=config.latent_dimension, out_channels=h.shape[-1], kernel_size=3,
                                padding=0),
                torch.nn.LeakyReLU(),
            ]
        self.cnn_backbone = torch.nn.Sequential(*cnn_layers)
        # self.embedding = MLP([self.input_dim, config.latent_dimension], norm=None)
        self.node_mlp_inner = MLP([self.input_dim, config.latent_dimension, config.latent_dimension],
                                  norm="layer_norm")
        self.node_mlp_outer = MLP([config.latent_dimension, config.latent_dimension, config.latent_dimension],
                                  norm="layer_norm")

    def forward(self, batch) -> torch.Tensor:
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(batch,
                                                                                                           remove_batch_dim=True)
        if self.z_pos_feature:
            z_pos = x[:, :, :, 2:3]
            z_pos = z_pos[context_trajs]
        if self.centralize_input:
            x = x - x[:, 0:1, :, :].mean(dim=2, keepdim=True)
        x = x[context_trajs]
        v = v[context_trajs]
        h = h[context_trajs]
        h = h[:, None, :, :].expand(-1, x.shape[1], -1, -1)
        batch_dim = x.shape[0]
        num_nodes = x.shape[2]
        num_ts = x.shape[1]
        if self.z_pos_feature:
            cnn_features = torch.cat([x, v, h, z_pos], dim=-1)
        else:
            cnn_features = torch.cat([x, v, h], dim=-1)
        cnn_features = cnn_features.permute(0, 2, 3, 1).reshape(-1, self.input_dim, num_ts)
        cnn_features = self.cnn_backbone(cnn_features)
        output_num_ts = cnn_features.shape[2]
        output = cnn_features.reshape(batch_dim, num_nodes, -1, output_num_ts).permute(0, 3, 1, 2)
        h = output[:, 0, :, :]
        # mesh pos and vel with h
        if self.z_pos_feature:
            input_vector = torch.cat([x[:, 0, :, :], v[:, 0, :, :], h, z_pos[:, 0, :, :]], dim=-1)
        else:
            input_vector = torch.cat([x[:, 0, :, :], v[:, 0, :, :], h], dim=-1)
        all_nodes_output = self.node_mlp_inner(input_vector)
        all_nodes_output = all_nodes_output.mean(dim=1)
        context_output = self.node_mlp_outer(all_nodes_output)
        # shape [num_context_trajs, latent_dim]

        context_output = torch.max(context_output, dim=0).values
        return context_output
