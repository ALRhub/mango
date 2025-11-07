import torch
from torch_geometric.utils import to_undirected


def get_edge_indices(face_indices, quad=True):
    """
    Get the edge indices from the face indices.
    Args:
        face_indices: Tensor of shape (num_faces, ?) containing the face indices
        quad: If True and face_indices contains 4 indices, it is interpreted as a quad, otherwise as a tetrahedron

    Returns: Tensor of shape (num_edges, 2) containing the edge indices
    """
    face_indices = torch.as_tensor(face_indices, dtype=torch.long)
    if face_indices.shape[1] == 3:
        edge_indices = torch.cat([face_indices[:, [0, 1]], face_indices[:, [1, 2]], face_indices[:, [2, 0]]], dim=0).T
    elif face_indices.shape[1] == 4:
        if quad:
            # quad
            edge_indices = torch.cat([face_indices[:, [0, 1]], face_indices[:, [1, 2]], face_indices[:, [2, 3]],
                                      face_indices[:, [3, 0]]], dim=0).T
        else:
            # tetrahedron
            edge_indices = torch.cat([face_indices[:, [0, 1]], face_indices[:, [1, 2]], face_indices[:, [2, 0]],
                                      face_indices[:, [0, 3]], face_indices[:, [1, 3]], face_indices[:, [2, 3]]], dim=0).T
    else:
        raise ValueError("Only triangles and quads/tetrahedrons are supported")
    edge_indices = to_undirected(edge_indices)
    return edge_indices
