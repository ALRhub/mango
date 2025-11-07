import torch
from torch_geometric.data import Batch, Data


def unpack_ml_batch(batch, remove_batch_dim=True):
    x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = batch["x"], batch["v"], batch["h"], batch["h_description"], batch[
        "edge_indices"], batch["edge_features"], batch["context_trajs"], batch["target_trajs"]
    if remove_batch_dim:
        assert x.shape[0] == 1, "Batch dimension must be 1"
        x = x[0]
        v = v[0]
        h = h[0]
        edge_indices = edge_indices[0]
        edge_features = edge_features[0]
        context_trajs = context_trajs[0]
        target_trajs = target_trajs[0]
        h_description = [h_value[0] for h_value in h_description]
    return x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs



def get_deformable_mask(graph: Batch | Data) -> torch.Tensor:
    return (graph.node_id[:, None] == graph.deformable_ids.view(1, -1)).any(dim=1)


def get_collider_mask(graph: Batch | Data) -> torch.Tensor:
    return (graph.node_id[:, None] == graph.collider_ids.view(1, -1)).any(dim=1)


def get_deformable_pos(batch_or_data):
    deformable_mask = get_deformable_mask(batch_or_data)
    return batch_or_data.pos[deformable_mask]
