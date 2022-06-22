import torch
from jaqpotpy.cfg import config
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GraphConv, GINConv, GINEConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0, norm: torch.nn.Module = None, act_first=True):
        super(GCN, self).__init__()
        torch.manual_seed(config.global_seed)

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        self.layers = torch.nn.ModuleList([GCNConv(in_channels, hidden_channels)])
        for i in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        # self.layers.append(torch.nn.ModuleList(GCNConv(hidden_channels, hidden_channels) for i in range(num_layers - 1)))
        self.out = torch.nn.Linear(hidden_channels, out_channels)
        # self.layers.append(torch.nn.ModuleList([torch.nn.Linear(hidden_channels, out_channels)]))

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


class GhebConv_V1(torch.nn.Module):

    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0, norm: torch.nn.Module = None, act_first=True):
        super(GhebConv_V1, self).__init__()
        torch.manual_seed(config.global_seed)

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        self.jaqpotpy_version = config.version

        self.layers = torch.nn.ModuleList([ChebConv(in_channels, hidden_channels)])
        for i in range(num_layers - 1):
            self.layers.append(ChebConv(hidden_channels, hidden_channels))
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


class GINConv_V1(torch.nn.Module):

    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0, norm: torch.nn.Module = None, act_first=True):
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


class SAGEConv_V1(torch.nn.Module):

    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0, norm: torch.nn.Module = None, act_first=True):
        super(SAGEConv_V1, self).__init__()
        torch.manual_seed(config.global_seed)

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        self.jaqpotpy_version = config.version

        self.layers = torch.nn.ModuleList([SAGEConv(in_channels, hidden_channels)])
        for i in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
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


class GCN_V1(torch.nn.Module):

    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0, norm: torch.nn.Module = None, act_first=True):
        super(GCN_V1, self).__init__()
        torch.manual_seed(config.global_seed)

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        self.jaqpotpy_version = config.version

        self.layers = torch.nn.ModuleList([GCNConv(in_channels, hidden_channels)])
        for i in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
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


class GraphConv_V1(torch.nn.Module):

    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0, norm: torch.nn.Module = None, act_first=True):
        super(GraphConv_V1, self).__init__()
        torch.manual_seed(config.global_seed)

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        self.jaqpotpy_version = config.version

        self.layers = torch.nn.ModuleList([GraphConv(in_channels, hidden_channels)])
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_channels, hidden_channels))
        # self.layers.append(torch.nn.ModuleList(GCNConv(hidden_channels, hidden_channels) for i in range(num_layers - 1)))
        self.out = torch.nn.Linear(hidden_channels, out_channels)
        # self.layers.append(torch.nn.ModuleList([torch.nn.Linear(hidden_channels, out_channels)]))

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