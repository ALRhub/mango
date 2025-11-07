import torch
from omegaconf import OmegaConf

from mango.dataset.util.graph_input_output_util import unpack_ml_batch
from mango.simulator.decoder import Decoder, ScaledTanh
from mango.simulator.gnn.egno.mgno import MGNO
from mango.simulator.util.mlp import MLP


class MGNDecoder(torch.nn.Module):

    def __init__(self, config, example_input_batch, eval_ds):
        super().__init__()
        self.config = config
        x = example_input_batch["x"]
        h = example_input_batch["h"]
        edge_features = example_input_batch["edge_features"]
        self.eval_ds = eval_ds
        self.z_pos_feature = config.z_pos_feature

        # although we will only use the first step and repeat that for decoding, the shape is the same for x and v
        world_dim = x.shape[-1]
        self.mgn_backbone = MGNO(
            n_layers=config.n_layers,
            h_dim=h.shape[-1] + config.context_encoding_dimension + 1 * self.z_pos_feature,  # concat h with context encoding for now, add z dim if enabled
            edge_feature_dim=edge_features.shape[-1],
            world_dim=world_dim,
            latent_dim=config.latent_dimension,
            activation=config.activation,
            scatter_reduce=config.scatter_reduce,
            use_hidden_layers=config.use_hidden_layers,
            use_time_conv=False,
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

    def eval_forward(self, batch, context_encoding):
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(batch,
                                                                                                           remove_batch_dim=True)
        x = x[target_trajs]
        v = v[target_trajs]
        h = h[target_trajs]
        edge_features = edge_features[target_trajs]
        # add context encoding to h tensor
        encoding = context_encoding.unsqueeze(0).unsqueeze(0).expand(h.shape[0], h.shape[1], -1)
        mgn_h = torch.cat((h, encoding), dim=-1)
        # take initial x/v state and repeat it for all time steps
        num_timesteps = x.shape[1]
        current_x = x[:, 0:1, :, :]
        current_v = v[:, 0:1, :, :]
        trajectory_length = x.shape[1]
        deformable_mask = h[0, :, 0] == 1
        output_trajectory = [torch.clone(current_x[:, :, deformable_mask, :])]
        for current_step in range(1, trajectory_length):  # last step does not need update
            def_vels = self.call_model(current_x, mgn_h, current_v, edge_indices, edge_features, deformable_mask)
            current_v = v[:, current_step:current_step + 1, :, :]
            current_v[:, :, deformable_mask, :] = def_vels
            next_x = x[:, current_step:current_step + 1, :, :]
            next_x[:, :, deformable_mask, :] = current_x[:, :, deformable_mask, :] + def_vels
            current_x = next_x
            output_trajectory.append(torch.clone(current_x[:, :, deformable_mask, :]))
        output_trajectory = torch.cat(output_trajectory, dim=1)
        return output_trajectory

    def train_forward(self, batch, encoding):
        if "context_trajs" in batch:
            # meta learning case
            x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(batch,
                                                                                                               remove_batch_dim=True)
            encoding = encoding.unsqueeze(0).unsqueeze(0).expand(h.shape[0], h.shape[1], -1)
            mgn_h = torch.cat((h, encoding), dim=-1)
            mgn_h = mgn_h[target_trajs]
            edge_features = edge_features[target_trajs]

            deformable_mask = h[0, :, 0] == 1
            x = x[target_trajs]
            # remove last step, since we don't have a label for that
            x = x[:, :-1, :, :]
            # have to noise the data since we don't use the step dataset
            noise = torch.randn_like(x) * self.config.noise_scale
            noisy_x = x + noise

            v = noisy_x[:, 1:, :, :] - noisy_x[:, :-1, :, :]
            # initial vel is zero
            v = torch.cat((torch.zeros((v.shape[0], 1, v.shape[2], v.shape[3])).to(v), v), dim=1)
            initial_def_pos = x[:, 0:1, deformable_mask, :]

            def_vels = self.call_model(noisy_x, mgn_h, v, edge_indices, edge_features, deformable_mask)
            prediction = noisy_x[:, :, deformable_mask, :] + def_vels
            prediction = torch.cat([initial_def_pos, prediction], dim=1)
            return prediction
        else:
            x, v, h, h_description, edge_indices, edge_features = (batch["x"], batch["v"],
                                                                   batch["h"], batch["h_description"],
                                                                   batch["edge_indices"][0], batch["edge_features"])
            # add context encoding to h tensor
            encoding = encoding.unsqueeze(0).unsqueeze(0).expand(h.shape[0], h.shape[1], -1)
            mgn_h = torch.cat((h, encoding), dim=-1)
            # add temporal dim to x and v
            x = x[:, None, :, :]
            v = v[:, None, :, :]
            deformable_mask = h[0, :, 0] == 1
            def_vels = self.call_model(x, mgn_h, v, edge_indices, edge_features, deformable_mask)
            prediction = x[:, :, deformable_mask, :] + def_vels
            # remove temporal dim
            prediction = prediction[:, 0, :, :]
            return prediction

    def call_model(self, x, h, v, edge_indices, edge_features, deformable_mask):
        if self.z_pos_feature:
            if x.shape[1] == 1:
                # mgn case or eval case without proper time dimension in batch
                h = torch.cat((h, x[:, 0, :, -1:]), dim=-1)
            else:
                # ml case with time dimension
                h = h[:, None, :, :].repeat(1, x.shape[1], 1, 1)
                h = torch.cat((h, x[:, :, :, -1:]), dim=-1)
        latent_prediction = self.mgn_backbone(x, h, v, edge_indices, edge_features)
        latent_prediction = latent_prediction[:, :, deformable_mask, :]
        vels = self.output_decoder(latent_prediction)
        return vels.to(torch.float32)

    def forward(self, batch, encoding) -> torch.Tensor:
        if self.training:
            return self.train_forward(batch, encoding)
        else:
            return self.eval_forward(batch, encoding)
