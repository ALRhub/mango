import torch

from mango.dataset.util.graph_input_output_util import unpack_ml_batch
from mango.dataset.util.util import add_noise
from mango.simulator.gnn.egno.basic import EGNN
from mango.simulator.gnn.egno.egno import EGNO

# This class is outdated and needs to be updated.
class EGNNDecoder(torch.nn.Module):

    def __init__(self, config, example_input_batch):
        super().__init__()
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(
            example_input_batch,
            remove_batch_dim=True)

        self.egnn_backbone = EGNN(
            n_layers=config.n_layers,
            in_node_nf=h.shape[-1] + config.context_encoding_dimension,  # concat h with context encoding for now
            in_edge_nf=edge_features.shape[-1],
            hidden_nf=config.latent_dimension,
            with_v=True,
        )
        self.config = config

    def train_forward(self, batch, encoding) -> torch.Tensor:
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(batch,
                                                                                                           remove_batch_dim=True)
        # add context encoding to h tensor
        encoding = encoding.unsqueeze(0).unsqueeze(0).repeat(h.shape[0], h.shape[1], 1)
        egnn_h = torch.cat((h, encoding), dim=-1)
        # repeat over all steps
        egnn_h = egnn_h.unsqueeze(1).repeat(1, x.shape[1], 1, 1)
        # take x/v states until the last step
        num_timesteps = x.shape[1]
        # collider nodes are already present in all time steps
        egnn_x = x[:, :-1, :, :]
        egnn_h = egnn_h[:, :-1, :, :]

        noise = torch.randn_like(egnn_x) * self.config.noise_scale
        noisy_egnn_x = egnn_x + noise

        v = noisy_egnn_x[:, 1:, :, :] - egnn_x[:, :-1, :, :]
        # initial vel is zero
        v = torch.cat((torch.zeros((v.shape[0], 1, v.shape[2], v.shape[3])).to(v), v), dim=1)

        predictions = []
        for traj in target_trajs:
            target_x = noisy_egnn_x[traj]
            target_h = egnn_h[traj]
            target_v = v[traj]
            pred_x, _, _ = self.egnn_backbone(target_x, target_h, edge_indices, edge_features, v=target_v)
            predictions.append(pred_x)
        predictions = torch.stack(predictions, dim=0)
        # add the initial state to the predictions
        target_x = x[target_trajs]
        predictions = torch.cat((target_x[:, :1, :, :], predictions), dim=1)
        # return only the deformable nodes -> ask h
        deformable_mask = h[0, :, 0] == 1
        return predictions[:, :, deformable_mask, :]

    def eval_forward(self, batch, encoding) -> torch.Tensor:
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(batch,
                                                                                                           remove_batch_dim=True)
        encoding = encoding.unsqueeze(0).unsqueeze(0).repeat(h.shape[0], h.shape[1], 1)
        egnn_h = torch.cat((h, encoding), dim=-1)
        # first step
        egnn_x = x[:, 0, :, :]
        egnn_v = v[:, 0, :, :]
        predictions = []
        for traj in target_trajs:
            target_x = egnn_x[traj]
            target_h = egnn_h[traj]
            target_v = egnn_v[traj]
            traj_pred = [target_x]
            for _ in range(x.shape[1] - 1):
                pred_x, _, _ = self.egnn_backbone(target_x, target_h, edge_indices, edge_features, v=target_v)
                traj_pred.append(pred_x)
                # compute new v
                target_v = pred_x - target_x
                target_x = pred_x
                # TODO: collider
            traj_pred = torch.stack(traj_pred, dim=0)
            predictions.append(traj_pred)
        predictions = torch.stack(predictions, dim=0)
        # return only the deformable nodes -> ask h
        deformable_mask = h[0, :, 0] == 1
        return predictions[:, :, deformable_mask, :]



    def forward(self, batch, encoding) -> torch.Tensor:
        if self.training:
            return self.train_forward(batch, encoding)
        else:
            return self.eval_forward(batch, encoding)
