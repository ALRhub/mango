from mango.simulator.gnn.egno.basic import EGNN
from mango.simulator.gnn.egno.layer_no import TimeConv, get_timestep_embedding, TimeConv_x
import torch.nn as nn
import torch


class EGNO(EGNN):
    def __init__(self, n_layers, in_node_nf, in_edge_nf, hidden_nf, activation=nn.SiLU(), with_v=True,
                 norm=False, use_time_conv=True, num_modes=2, num_timesteps=8, time_emb_dim=32, update_vel=True):
        self.time_emb_dim = time_emb_dim
        in_node_nf = in_node_nf + self.time_emb_dim

        super(EGNO, self).__init__(n_layers, in_node_nf, in_edge_nf, hidden_nf, activation, with_v,  norm)
        self.use_time_conv = use_time_conv
        self.num_timesteps = num_timesteps
        self.hidden_nf = hidden_nf

        if use_time_conv:
            self.time_conv_modules = nn.ModuleList()
            self.time_conv_x_modules = nn.ModuleList()
            for i in range(n_layers):
                self.time_conv_modules.append(TimeConv(hidden_nf, hidden_nf, num_modes, activation, with_nin=False))
                self.time_conv_x_modules.append(TimeConv_x(2, 2, num_modes, activation, with_nin=False))

    def forward(self, x, h, edge_index, edge_fea, v=None):
        """
        :param x: shape [num_ts, num_nodes, world_dim]
        :param h:  [num_nodex, node_feature_dim]
        :param edge_index:
        :param edge_fea:
        :param v:
        :return:
        """
        # [BN, H]
        world_dim = x.shape[-1]
        assert world_dim == 3
        T = self.num_timesteps

        num_nodes = h.shape[0]
        num_edges = edge_index[0].shape[0]

        cumsum = torch.arange(0, T).to(x.device) * num_nodes
        # cumsum_nodes = cumsum.repeat_interleave(num_nodes, dim=0)
        cumsum_edges = cumsum.repeat_interleave(num_edges, dim=0)

        time_emb = get_timestep_embedding(torch.arange(T).to(x), embedding_dim=self.time_emb_dim,
                                          max_positions=10000)  # [T, H_t]
        h = h.unsqueeze(0).repeat(T, 1, 1)  # [T, BN, H]
        time_emb = time_emb.unsqueeze(1).repeat(1, num_nodes, 1)  # [T, BN, H_t]
        h = torch.cat((h, time_emb), dim=-1)  # [T, BN, H+H_t]
        h = h.view(-1, h.shape[-1])  # [T*BN, H+H_t]

        h = self.embedding(h)
        loc_mean = x.mean(dim=1, keepdim=True).repeat(1, x.shape[1], 1)

        x = x.view(T * num_nodes, world_dim)
        loc_mean = loc_mean.view(T * num_nodes, world_dim)
        v = v.view(T * num_nodes, world_dim)

        edges_0 = edge_index[0].repeat(T) + cumsum_edges
        edges_1 = edge_index[1].repeat(T) + cumsum_edges
        edge_index = [edges_0, edges_1]
        edge_fea = edge_fea.repeat(T, 1)

        for i in range(self.n_layers):
            if self.use_time_conv:
                time_conv = self.time_conv_modules[i]
                h = time_conv(h.view(T, num_nodes, self.hidden_nf)).view(T * num_nodes, self.hidden_nf)
                x_translated = x - loc_mean
                time_conv_x = self.time_conv_x_modules[i]
                X = torch.stack((x_translated, v), dim=-1)
                temp = time_conv_x(X.view(T, num_nodes, world_dim, 2))
                x = temp[..., 0].view(T * num_nodes, world_dim) + loc_mean
                v = temp[..., 1].view(T * num_nodes, world_dim)

            x, v, h = self.layers[i](x, h, edge_index, edge_fea, v=v)
        return (x, v, h) if v is not None else (x, h)



