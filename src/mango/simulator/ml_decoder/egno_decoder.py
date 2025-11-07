import torch

from mango.dataset.util.graph_input_output_util import unpack_ml_batch
from mango.simulator.gnn.egno.egno import EGNO

# This class is outdated and needs to be updated.
class EGNODecoder(torch.nn.Module):

    def __init__(self, config, example_input_batch):
        super().__init__()
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(example_input_batch,
                                                                              remove_batch_dim=True)
        # although we will only use the first step and repeat that for decoding, the shape is the same for x and v
        self.egno_backbone = EGNO(
            n_layers=config.n_layers,
            in_node_nf=h.shape[-1] + config.context_encoding_dimension,  # concat h with context encoding for now
            in_edge_nf=edge_features.shape[-1],
            hidden_nf=config.latent_dimension,
            use_time_conv=True,
            num_timesteps=x.shape[1],
            time_emb_dim=config.time_emb_dim
        )

    def forward(self, batch, encoding) -> torch.Tensor:
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(batch, remove_batch_dim=True)
        original_2_dim = x.shape[-1] == 2
        if original_2_dim:
            x = torch.cat((x, torch.zeros_like(x[:, :, :, 0:1])), dim=-1)
            v = torch.cat((v, torch.zeros_like(v[:, :, :, 0:1])), dim=-1)

        # add context encoding to h tensor
        encoding = encoding.unsqueeze(0).unsqueeze(0).repeat(h.shape[0], h.shape[1], 1)
        egno_h = torch.cat((h, encoding), dim=-1)
        # take initial x/v state and repeat it for all time steps
        num_timesteps = x.shape[1]
        init_x = x[:, 0:1, :, :]
        init_v = v[:, 0:1, :, :]
        # add third dimension if missing
        egno_x = init_x.repeat(1, num_timesteps, 1, 1)
        egno_v = init_v.repeat(1, num_timesteps, 1, 1)
        # collider nodes should be present in all time steps
        last_index = len(h_description) - 1 - h_description[::-1].index("one_hot")
        if last_index == 2:
            collider_index = 1
            # deformable, collider, encoder_node
            collider_mask = h[0, :, collider_index] == 1
            collider_nodes = x[:, :, collider_mask, :]
            egno_x[:, :, collider_mask, :] = collider_nodes
            collider_v = v[:, :, collider_mask, :]
            egno_v[:, :, collider_mask, :] = collider_v
        if last_index > 2:
            # something not implemented yet
            raise NotImplementedError("More than 3 node types detected.")
        # predict all trajectories
        predictions = []
        for traj_idx in target_trajs:
            target_x = egno_x[traj_idx]
            target_v = egno_v[traj_idx]
            target_h = egno_h[traj_idx]
            target_edge_features = edge_features[traj_idx]
            num_timesteps = target_x.shape[0]
            num_nodes = target_x.shape[1]
            world_dim = target_x.shape[-1]
            pred_x, _, _ = self.egno_backbone(target_x, target_h, edge_indices, target_edge_features, v=target_v)
            pred_x = pred_x.view(num_timesteps, num_nodes, world_dim)
            predictions.append(pred_x)
        predictions = torch.stack(predictions, dim=0)
        # reduce dim
        if original_2_dim:
            predictions = predictions[:, :, :, 0:2]
        # return only the deformable nodes -> ask h
        deformable_mask = h[0, :, 0] == 1
        return predictions[:, :, deformable_mask, :]

