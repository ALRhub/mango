from abc import abstractmethod, ABC
import torch

from mango.dataset.util.graph_input_output_util import unpack_ml_batch


class AbstractEncoder(torch.nn.Module, ABC):

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def add_encoder_output_node(self, batch):
        """
        Adds a node to the graph which is connected to all other nodes, but only 1 way:
        The information can only pass to the node, but no information can be passed from it.
        :param batch: dict containing the graph information
        :return: updated dict
        """
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(batch, remove_batch_dim=True)
        edge_feature_description = batch["edge_feature_description"]
        edge_feature_description = [desc[0] for desc in edge_feature_description]
        # add encoder node to all graphs, with zero position
        x = torch.cat([x, torch.zeros_like(x[:, :, 0:1, :])], dim=2)
        v = torch.cat([v, torch.zeros_like(v[:, :, 0:1, :])], dim=2)
        h = torch.cat([h, torch.zeros_like(h[:, 0:1, :])], dim=1)
        last_index = len(h_description) - 1 - h_description[::-1].index("one_hot")
        h[:, -1, last_index] = 1.0
        # add edge from all nodes to the last node
        num_nodes = x.shape[2] - 1
        # add edges from all nodes to the encoder node, for some reason, this means the row indices are the encoder node
        new_edges = torch.stack(
            [torch.ones(num_nodes, dtype=torch.int32).to(edge_indices) * num_nodes,
             torch.arange(num_nodes, dtype=torch.int32).to(edge_indices)], dim=0)
        edge_indices = torch.cat([edge_indices, new_edges], dim=1)
        edge_features = torch.cat([edge_features, torch.zeros(x.shape[0], num_nodes, edge_features.shape[-1]).to(edge_features)], dim=1)
        last_edge_index = len(edge_feature_description) - 1 - edge_feature_description[::-1].index("one_hot")
        edge_features[:, -num_nodes:, last_edge_index] = 1.0
        return x, v, h, h_description, edge_indices, edge_features, context_trajs






