from torch_geometric.data import Batch, Data
from torch_geometric import transforms


def add_distances_from_positions(data_or_batch: Batch | Data, add_euclidian_distance: bool) -> Batch | Data:
    """
    Transform the node positions to the edges as relative distance together with (if needed) Euclidean norm and add
    them to the edge features
    :param data_or_batch:
    :return:
    """

    def _update_edge_type_description(edge_feature_description, add_z_distance: bool, add_euclidian_distance: bool):
        edge_feature_description.extend(["x_distance", "y_distance"])
        if add_z_distance:
            edge_feature_description.append("z_distance")
        if add_euclidian_distance:
            edge_feature_description.append("euclidian_distance")

    if data_or_batch.edge_index is None or data_or_batch.edge_index.shape[1] == 0:
        # there are no edges, so we can't add edge features. Do nothing in this case
        return data_or_batch

    if hasattr(data_or_batch, "edge_feature_description"):
        add_z_distance = data_or_batch.pos.shape[1] == 3
        if isinstance(data_or_batch, Batch):
            for edge_feature_description in data_or_batch.edge_feature_description:
                _update_edge_type_description(edge_feature_description=edge_feature_description,
                                              add_z_distance=add_z_distance,
                                              add_euclidian_distance=add_euclidian_distance)
        else:
            _update_edge_type_description(edge_feature_description=data_or_batch.edge_feature_description,
                                          add_z_distance=add_z_distance,
                                          add_euclidian_distance=add_euclidian_distance)

    if add_euclidian_distance:
        data_transform = transforms.Compose([transforms.Cartesian(norm=False, cat=True),
                                             transforms.Distance(norm=False, cat=True)])
    else:
        data_transform = transforms.Cartesian(norm=False, cat=True)
    out_batch = data_transform(data_or_batch)
    return out_batch
