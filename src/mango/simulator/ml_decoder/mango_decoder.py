import torch
from omegaconf import OmegaConf

from mango.dataset.util.graph_input_output_util import unpack_ml_batch
from mango.simulator.decoder import Decoder, ScaledTanh
from mango.simulator.mango import Mango
from mango.simulator.util.mlp import MLP


class MangoDecoder(torch.nn.Module):

    def __init__(self, config, example_input_batch):
        super().__init__()
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(
            example_input_batch,
            remove_batch_dim=True)
        self.z_pos_feature = config.z_pos_feature
        # although we will only use the first step and repeat that for decoding, the shape is the same for x and v
        world_dim = x.shape[-1]
        self.mgno_backbone = Mango(
            n_layers=config.n_layers,
            h_dim=h.shape[-1] + config.context_encoding_dimension + 1 * self.z_pos_feature,  # concat h with context encoding for now
            edge_feature_dim=edge_features.shape[-1],
            world_dim=world_dim,
            latent_dim=config.latent_dimension,
            activation=config.activation,
            time_emb_dim=config.time_emb_dim,
            scatter_reduce=config.scatter_reduce,
            use_hidden_layers=config.use_hidden_layers,
            use_time_conv=True,
            time_conv_type=config.time_conv_type,
        )
        
        decode_module = MLP(in_features=config.latent_dimension,
                            latent_dimension=config.output_decoder.latent_dimension,
                            config=OmegaConf.create(dict(activation_function="relu",
                                                         add_output_layer=False,
                                                         num_layers=1,
                                                         regularization={
                                                             "dropout": config.output_decoder.regularization.dropout,
                                                         },
                                                         )),
                            )
        readout_module = torch.nn.Linear(config.output_decoder.latent_dimension, world_dim)
        if config.output_decoder.tanh.enabled:
            output_activation = ScaledTanh(config.output_decoder.tanh.scale_factor,
                                           config.output_decoder.tanh.input_scale_factor)
        else:
            output_activation = torch.nn.Identity()
        self.output_decoder = Decoder(decode_module, readout_module, output_activation)

    def forward(self, batch, encoding) -> torch.Tensor:
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(batch,
                                                                                                           remove_batch_dim=True)
        # add context encoding to h tensor
        encoding = encoding.unsqueeze(0).unsqueeze(0).expand(h.shape[0], h.shape[1], -1)
        mgno_h = torch.cat((h, encoding), dim=-1)
        # take initial x/v state and repeat it for all time steps
        num_timesteps = x.shape[1]
        init_x = x[:, 0:1, :, :]
        init_v = v[:, 0:1, :, :]
        mgno_x = init_x.repeat(1, num_timesteps, 1, 1)
        mgno_v = init_v.repeat(1, num_timesteps, 1, 1)
        # collider nodes should be present in all time steps
        last_index = len(h_description) - 1 - h_description[::-1].index("one_hot")
        if last_index == 2:
            collider_index = 1
            # deformable, collider, encoder_node
            collider_mask = h[0, :, collider_index] == 1
            collider_nodes = x[:, :, collider_mask, :]
            mgno_x[:, :, collider_mask, :] = collider_nodes
            collider_v = v[:, :, collider_mask, :]
            mgno_v[:, :, collider_mask, :] = collider_v
        if last_index > 2:
            # something not implemented yet
            raise NotImplementedError("More than 3 node types detected.")
        # predict all trajectories
        mgno_x = mgno_x[target_trajs]
        mgno_v = mgno_v[target_trajs]
        mgno_h = mgno_h[target_trajs]
        edge_features = edge_features[target_trajs]
        if self.z_pos_feature:
            # ml case with time dimension
            mgno_h = mgno_h[:, None, :, :].repeat(1, x.shape[1], 1, 1)
            mgno_h = torch.cat((mgno_h, mgno_x[:, :, :, -1:]), dim=-1)
        latent_prediction = self.mgno_backbone(mgno_x, mgno_h, mgno_v, edge_indices, edge_features)
        displacements = self.output_decoder(latent_prediction)
        # add displacements to initial prediction, but only for deformable nodes
        deformable_mask = h[0, :, 0] == 1
        predictions = mgno_x[:, :, deformable_mask, :] + displacements[:, :, deformable_mask, :]
        return predictions

