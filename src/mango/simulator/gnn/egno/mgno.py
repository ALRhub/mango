from mango.simulator.gnn.egno.basic import BaseMLP
from mango.simulator.gnn.egno.layer_no import get_timestep_embedding
import torch.nn as nn
import torch


class MGNOTimeConv(nn.Module):
    def __init__(self, in_ch, out_ch, modes, act):
        super().__init__()
        self.t_conv = MGNOSpectralConv1d(in_ch, out_ch, modes)
        self.act = act

    def forward(self, x, message):
        h = self.t_conv(x)
        out = self.act(h)
        return x + out, message


@torch.jit.script
def compl_mul1d_x(a, b):
    # (batch_size, modes, num_nodes, latent_dim), (latent_dim (input), latent_dim (output), modes) -> (batch_size, modes, num_nodes, latent_dim)
    return torch.einsum("bmni,iom->bmno", a, b)


class MGNOSpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1):
        super().__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1

        self.scale = (1 / (in_ch * out_ch))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, self.modes1, 2, dtype=torch.float))

    def forward(self, x):
        batch_size, T, num_nodes, latent_dim = x.shape  # D should be 3
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        with torch.cuda.amp.autocast(enabled=False):
            # with torch.autocast(device_type='cuda', enabled=False):
            x_ft = torch.fft.rfftn(x.float(), dim=[1])
            # Multiply relevant Fourier modes
            out_ft = compl_mul1d_x(x_ft[:, :self.modes1], torch.view_as_complex(self.weights1))
            # Return to physical space
            x = torch.fft.irfftn(out_ft, s=[T], dim=[1])
        return x


class MGNOTimeConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, padding=3, activation=nn.LeakyReLU(), residual=True,
                 pooling=False):
        super().__init__()
        if residual:
            assert in_ch == out_ch, "Residual connection requires in_ch == out_ch"
        self.residual = residual
        self.t_conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.pooling = pooling
        if self.pooling:
            self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.act = activation

    def forward(self, x, edge_message):
        # x has shape [batch_dim, num_ts, num_nodes, hidden_dim]
        # Want to apply 1D convolution over the time dimension, batch_dim, and num_nodes are batch dims
        # output shape should be [batch_dim*num_nodes, hidden_dim, num_ts]
        batch_dim = x.shape[0]
        num_nodes = x.shape[2]
        hidden_dim = x.shape[-1]
        num_ts = x.shape[1]

        # for pooling
        num_edges = edge_message.shape[2]

        h = x.permute(0, 2, 3, 1).reshape(-1, hidden_dim, num_ts)
        h = self.t_conv(h)
        h = self.act(h)
        output_num_ts = h.shape[2]
        out = h.reshape(batch_dim, num_nodes, -1, output_num_ts).permute(0, 3, 1, 2)

        if self.residual:
            x = x + out
        else:
            x = out
        # have to do pooling here, since otherwise the residual won't work
        if self.pooling:
            h = x.permute(0, 2, 3, 1).reshape(-1, hidden_dim, num_ts)
            edge_message = edge_message.permute(0, 2, 3, 1).reshape(-1, hidden_dim, num_ts)
            h = self.pooling(h)
            edge_message = self.pooling(edge_message)
            output_num_ts = h.shape[2]
            out = h.reshape(batch_dim, num_nodes, -1, output_num_ts).permute(0, 3, 1, 2)
            edge_message = edge_message.reshape(batch_dim, num_edges, -1, output_num_ts).permute(0, 3, 1, 2)
        return out, edge_message


@torch.compile
def mgno_aggregate(message, row, num_nodes, aggr="sum"):
    """
    Batched aggregation method
    :param message: shape [batch_dim, num_ts, num_edges, hidden_dim]
    :param row: shape [num_edges]
    :param aggr: "sum" or "mean"
    :return: aggretaged message shape [batch_dim, num_ts, num_nodes, hidden_dim]
    """
    batch_dim, num_ts, num_edges, hidden_dim = message.shape
    result_shape = (batch_dim, num_ts, num_nodes, hidden_dim)
    result = torch.zeros(result_shape).to(message.device)
    row = row[None, None, :, None].expand(batch_dim, num_ts, -1, hidden_dim)
    result.scatter_add_(2, row, message)
    if aggr == "sum":
        pass
    elif aggr == "mean":
        count = torch.zeros(result_shape).to(message.device)
        ones = torch.ones_like(message)
        count.scatter_add_(2, row, ones)
        result = result / count.clamp(min=1)
    else:
        raise ValueError(f"Unknown aggregation method: {aggr}")
    return result


class MGNO(torch.nn.Module):
    def __init__(self, n_layers, h_dim, edge_feature_dim, world_dim, latent_dim, activation="leakyrelu",
                 use_hidden_layers=False, dropout=0.0,
                 num_modes=2, time_emb_dim=32, scatter_reduce="mean", use_time_conv=True, time_conv_type="spectral"):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.world_dim = world_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.scatter_reduce = scatter_reduce
        self.use_hidden_layers = use_hidden_layers
        self.dropout = dropout

        self.node_update_modules = nn.ModuleList()
        self.node_layer_norm_modules = nn.ModuleList()
        self.edge_update_modules = nn.ModuleList()
        self.edge_layer_norm_modules = nn.ModuleList()

        # if no time_conv: This is just a standard MGN
        self.use_time_conv = use_time_conv
        if use_time_conv:
            self.time_conv_modules = nn.ModuleList()

        # embeddings
        if use_time_conv:
            self.node_embedding = nn.Linear(h_dim + world_dim + time_emb_dim, latent_dim)
            self.edge_embedding = nn.Linear(edge_feature_dim + world_dim + time_emb_dim, latent_dim)
        else:
            self.node_embedding = nn.Linear(h_dim + world_dim, latent_dim)
            self.edge_embedding = nn.Linear(edge_feature_dim + world_dim, latent_dim)
        if self.dropout > 0:
            self.node_embedding = nn.Sequential(self.node_embedding, nn.Dropout(p=self.dropout))
            self.edge_embedding = nn.Sequential(self.edge_embedding, nn.Dropout(p=self.dropout))

        if self.use_hidden_layers:
            hidden_dim = latent_dim
        else:
            hidden_dim = -1

        if activation == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation == "silu":
            activation = nn.SiLU()
        elif activation == "relu":
            activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

        for i in range(n_layers):
            self.node_update_modules.append(
                BaseMLP(2 * latent_dim, hidden_dim, latent_dim, activation, residual=False, last_act=False))
            self.node_layer_norm_modules.append(nn.LayerNorm(latent_dim))
            self.edge_update_modules.append(
                BaseMLP(3 * latent_dim, hidden_dim, latent_dim, activation, residual=False, last_act=False))
            self.edge_layer_norm_modules.append(nn.LayerNorm(latent_dim))

        if use_time_conv:
            if time_conv_type == "spectral":
                for i in range(n_layers):
                    self.time_conv_modules.append(MGNOTimeConv(latent_dim, latent_dim, num_modes, act=nn.LeakyReLU()))
            elif time_conv_type == "conv_decoder":
                for i in range(n_layers):
                    self.time_conv_modules.append(
                        MGNOTimeConv1d(latent_dim, latent_dim, kernel_size=7, padding=3, activation=activation,
                                       pooling=False))
            elif time_conv_type == "conv_encoder":
                pooling_layers = [1, 3, 5, 7, 9]
                for i in range(n_layers):
                    self.time_conv_modules.append(
                        MGNOTimeConv1d(latent_dim, latent_dim, kernel_size=3, padding=1, activation=activation,
                                       pooling=i in pooling_layers))

    def forward(self, x, h, v, edge_index, edge_fea):
        """
        :param x: shape [batch_dim, num_ts, num_nodes, world_dim]
        :param h: shape [batch_dim, num_nodes, h_dim] or [batch_dim, num_ts, num_nodes, h_dim]
        :param v: shape [batch_dim, num_ts, num_nodes, world_dim]
        :param edge_index: shape [2, num_edges]  (consistent over batch and time)
        :param edge_fea: shape [batch_dim, num_edges, edge_feature_dim] (consistent and time)

        :return:
        """
        # check for repeating of h
        if len(h.shape) == 3:
            h = h.unsqueeze(1).repeat(1, x.shape[1], 1, 1)
        # repeat edge_fea over time dim
        edge_fea = edge_fea[:, None, :, :].repeat(1, x.shape[1], 1, 1)

        # all inputs (except edge_index) are now [batch_dim, num_ts, num_nodes/num_edges, feature_dim]

        # Embeddings
        row, col = edge_index
        r_ij = x[:, :, row, :] - x[:, :, col, :]  # relative position
        if self.use_time_conv:
            T = x.shape[1]
            time_emb = get_timestep_embedding(torch.arange(T).to(x), embedding_dim=self.time_emb_dim,
                                              max_positions=10000)  # shape (T, time_emb_dim)
            node_time_emb = time_emb[None, :, None, :].expand(x.shape[0], -1, x.shape[2],
                                                              -1)  # shape (batch_dim, T, num_nodes, time_emb_dim)
            edge_time_emb = time_emb[None, :, None, :].expand(x.shape[0], -1, edge_index.shape[1],
                                                              -1)  # shape (batch_dim, T, num_edges, time_emb_dim)
            h = torch.cat([h, v, node_time_emb], dim=-1)
            message = torch.cat([edge_fea, r_ij, edge_time_emb], dim=-1)
        else:
            h = torch.cat([h, v], dim=-1)
            message = torch.cat([edge_fea, r_ij], dim=-1)

        h = self.node_embedding(h)
        message = self.edge_embedding(message)

        num_nodes = x.shape[2]

        # Processing
        for i in range(self.n_layers):
            # Time Convolution
            if self.use_time_conv:
                time_conv = self.time_conv_modules[i]
                h, message = time_conv(h, message)

            # Edge Update
            edge_update = self.edge_update_modules[i]
            edge_ln = self.edge_layer_norm_modules[i]
            # residual MLP
            message = edge_ln(edge_update(torch.cat([message, h[:, :, row, :], h[:, :, col, :]], dim=-1))) + message

            # Node Update
            aggr_message = mgno_aggregate(message, row, num_nodes=num_nodes, aggr=self.scatter_reduce)
            node_update = self.node_update_modules[i]
            node_ln = self.node_layer_norm_modules[i]
            # residual MLP
            h = node_ln(node_update(torch.cat([h, aggr_message], dim=-1))) + h
        return h
