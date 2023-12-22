from jaqpotpy.cfg import config
from torch_geometric.typing import OptTensor, Adj
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GINConv\
    , GINEConv, GraphConv, GatedGraphConv, ResGatedGraphConv, GATConv, GatedGraphConv
from torch_geometric.nn import global_mean_pool, global_add_pool,GATConv, MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter



class GCN_V1(torch.nn.Module):
    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0.0, norm: torch.nn.Module = None, act_first=True):
        super(GCN_V1, self).__init__()
        torch.manual_seed(config.global_seed)

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        # self.jaqpotpy_version = config.version

        self.layers = torch.nn.ModuleList([GCNConv(in_channels, hidden_channels).jittable()])
        for i in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_channels, hidden_channels).jittable())
        self.out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            if self.norm is not None and self.act_first is True:
                x = self.act(x)
                x = self.norm(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            elif self.norm is not None and self.act_first is False:
                x = self.norm(x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.out(x)
        return x


class GhebConvV1(torch.nn.Module):

    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels, k=10
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0.0,  norm: torch.nn.Module = None, act_first=True):
        super(GhebConvV1, self).__init__()
        torch.manual_seed(config.global_seed)

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        self.layers = torch.nn.ModuleList([ChebConv(in_channels, hidden_channels, K=k).jittable()])
        for i in range(num_layers - 1):
            self.layers.append(ChebConv(hidden_channels, hidden_channels, K=k).jittable())
        self.out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):

        for layer in self.layers:
            x = layer(x, edge_index)
            if self.norm is not None and self.act_first is True:
                x = self.act(x)
                x = self.norm(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            elif self.norm is not None and self.act_first is False:
                x = self.norm(x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.out(x)
        return x


class SAGEConv_V1(torch.nn.Module):

    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0.0, norm: torch.nn.Module = None, act_first=True):
        super(SAGEConv_V1, self).__init__()
        torch.manual_seed(config.global_seed)

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        self.layers = torch.nn.ModuleList([SAGEConv(in_channels, hidden_channels).jittable()])
        for i in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels).jittable())
        self.out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            if self.norm is not None and self.act_first is True:
                x = self.act(x)
                x = self.norm(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            elif self.norm is not None and self.act_first is False:
                x = self.norm(x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.out(x)
        return x


class GraphConv_V1(torch.nn.Module):

    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0.0, norm: torch.nn.Module = None, act_first=True):
        super(GraphConv_V1, self).__init__()
        torch.manual_seed(config.global_seed)

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        self.layers = torch.nn.ModuleList([GraphConv(in_channels, hidden_channels).jittable()])
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_channels, hidden_channels).jittable())
        self.out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            if self.norm is not None and self.act_first is True:
                x = self.act(x)
                x = self.norm(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            elif self.norm is not None and self.act_first is False:
                x = self.norm(x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.out(x)
        return x


class GatedGraphConv_V1(torch.nn.Module):

    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0.0, norm: torch.nn.Module = None, act_first=True):
        super(GatedGraphConv_V1, self).__init__()
        torch.manual_seed(config.global_seed)

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        self.layers = torch.nn.ModuleList([GraphConv(in_channels, hidden_channels).jittable()])
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_channels, hidden_channels).jittable())
        self.out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            if self.norm is not None and self.act_first is True:
                x = self.act(x)
                x = self.norm(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            elif self.norm is not None and self.act_first is False:
                x = self.norm(x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.out(x)
        return x


class ResGatedGraphConv_V1(torch.nn.Module):

    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0.0, norm: torch.nn.Module = None, act_first=True):
        super(ResGatedGraphConv_V1, self).__init__()
        torch.manual_seed(config.global_seed)

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        self.layers = torch.nn.ModuleList([ResGatedGraphConv(in_channels, hidden_channels).jittable()])
        for i in range(num_layers - 1):
            self.layers.append(ResGatedGraphConv(hidden_channels, hidden_channels).jittable())
        self.out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            if self.norm is not None and self.act_first is True:
                x = self.act(x)
                x = self.norm(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            elif self.norm is not None and self.act_first is False:
                x = self.norm(x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.out(x)
        return x


class GATConv_V1(torch.nn.Module):

    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0.0, norm: torch.nn.Module = None, act_first=True):
        super(GATConv_V1, self).__init__()
        torch.manual_seed(config.global_seed)

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        self.layers = torch.nn.ModuleList([GATConv(in_channels, hidden_channels).jittable()])
        for i in range(num_layers - 1):
            self.layers.append(GATConv(hidden_channels, hidden_channels).jittable())
        self.out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            if self.norm is not None and self.act_first is True:
                x = self.act(x)
                x = self.norm(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            elif self.norm is not None and self.act_first is False:
                x = self.norm(x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.out(x)
        return x


class AttentiveFPModel_V1(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, edge_dim: int, num_layers: int,
                 num_timesteps: int, dropout: float = 0.0):
        super().__init__()

        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        conv = GATConv(hidden_channels, hidden_channels, edge_dim).jittable()
        gru = GatedGraphConv(hidden_channels, hidden_channels).jittable()
        self.atom_convs = torch.nn.ModuleList([conv])
        self.atom_grus = torch.nn.ModuleList([gru])
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01).jittable()
            self.atom_convs.append(conv)
            self.atom_grus.append(GatedGraphConv(hidden_channels, hidden_channels).jittable())

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01).jittable()
        self.mol_gru = GatedGraphConv(hidden_channels, hidden_channels).jittable()

        self.lin2 = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch):
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.atom_convs[0](x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.atom_grus[0](h, x).relu_()

        for conv, gru in zip(self.atom_convs[1:], self.atom_grus[1:]):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin2(out)


class GATEConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.Tensor(1, out_channels))
        self.att_r = Parameter(torch.Tensor(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out + self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j * self.att_l).sum(dim=-1)
        alpha_i = (x_i * self.att_r).sum(dim=-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1)


class AttentiveFP(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
                                  dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                batch: Tensor) -> Tensor:
        """"""
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin2(out)

    def jittable(self) -> 'AttentiveFP':
        self.gate_conv = self.gate_conv.jittable()
        self.atom_convs = torch.nn.ModuleList(
            [conv.jittable() for conv in self.atom_convs])
        self.mol_conv = self.mol_conv.jittable()
        return self

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'edge_dim={self.edge_dim}, '
                f'num_layers={self.num_layers}, '
                f'num_timesteps={self.num_timesteps}'
                f')')





















class GINConv_V1(torch.nn.Module):

    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0.0, norm: torch.nn.Module = None, act_first=True):
        super(GINConv_V1, self).__init__()
        torch.manual_seed(config.global_seed)

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        self.jaqpotpy_version = config.version

        self.layers = torch.nn.ModuleList([GINConv(in_channels, hidden_channels)])
        for i in range(num_layers - 1):
            self.layers.append(GINConv(hidden_channels, hidden_channels))
        self.out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
            x = layer(x, edge_index)
            if self.norm and self.act_first is True:
                x = self.act(x)
                x = self.norm(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            elif self.norm and self.act_first is False:
                x = self.norm(x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, data.batch)
        x = self.out(x)
        return x


# class GCN_V1(torch.nn.Module):
#
#     def __init__(self, in_channels
#                  , num_layers, hidden_channels, out_channels
#                  , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0, norm: torch.nn.Module = None, act_first=True):
#         super(GCN_V1, self).__init__()
#         torch.manual_seed(config.global_seed)
#
#         self.act = activation
#         self.dropout = dropout
#         self.act_first = act_first
#         self.norm = norm
#
#         self.layers = torch.nn.ModuleList([GCNConv(in_channels, hidden_channels)])
#         for i in range(num_layers - 1):
#             self.layers.append(GCNConv(hidden_channels, hidden_channels))
#         self.out = torch.nn.Linear(hidden_channels, out_channels)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         for layer in self.layers:
#             x = layer(x, edge_index)
#             if self.norm and self.act_first is True:
#                 x = self.act(x)
#                 x = self.norm(x)
#                 x = F.dropout(x, p=self.dropout, training=self.training)
#             elif self.norm and self.act_first is False:
#                 x = self.norm(x)
#                 x = self.act(x)
#                 x = F.dropout(x, p=self.dropout, training=self.training)
#             else:
#                 x = self.act(x)
#                 x = F.dropout(x, p=self.dropout, training=self.training)
#         x = global_mean_pool(x, data.batch)
#         x = self.out(x)
#         return x


class GCN_t(torch.nn.Module):
    def __init__(self, graph_layers, linear_layers):
        super(GCN_t, self).__init__()
        torch.manual_seed(config.global_seed)
        self.graph_layers = graph_layers
        self.linear_layers = linear_layers
        # self.conv1 = GCNConv(30, 40)
        # self.conv2 = GCNConv(40, 40)
        # self.conv3 = GCNConv(40, 40)
        # self.lin = Linear(40, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.graph_layers(x, edge_index)
        x = self.linear_layers(x)

        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        # x = global_mean_pool(x, data.batch)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin(x)

        return x


class GATEConv_V1(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.Tensor(1, out_channels))
        self.att_r = Parameter(torch.Tensor(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: torch.Tensor, edge_index: Adj, edge_attr: torch.Tensor) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = out + self.bias
        return out

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, edge_attr: torch.Tensor,
                index: torch.Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:

        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j * self.att_l).sum(dim=-1)
        alpha_i = (x_i * self.att_r).sum(dim=-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1)


class AttentiveFP_V1(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, edge_dim: int, num_layers: int,
                 num_timesteps: int, dropout: float = 0.0):
        super().__init__()

        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        conv = GATEConv_V1(hidden_channels, hidden_channels, edge_dim, dropout)
        gru = GRUCell(hidden_channels, hidden_channels)
        self.atom_convs = torch.nn.ModuleList([conv])
        self.atom_grus = torch.nn.ModuleList([gru])
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        """"""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.atom_convs[0](x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.atom_grus[0](h, x).relu_()

        for conv, gru in zip(self.atom_convs[1:], self.atom_grus[1:]):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin2(out)


