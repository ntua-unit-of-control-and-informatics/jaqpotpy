"""Pytorch Utilities.
This file contains utilities that compute Datastractures adn other for Pytorch.
"""

import numpy as np


def to_torch_graph_data_and_y(g, y):
    """Returns torch graph data from jaqpot graph_data and adds y"""
    import torch
    from torch_geometric.data import Data

    torch_graph = Data(
        x=torch.FloatTensor(g.node_features),
        edge_index=torch.LongTensor(g.edge_index),
        edge_attr=g.edge_features,
        num_nodes=g.num_nodes,
        y=np.array([y]),
    )
    return torch_graph


def to_torch_graph_data(g):
    """Returns torch graph data from jaqpot graph_data without endpoint"""
    import torch
    from torch_geometric.data import Data

    torch_graph = Data(
        x=torch.FloatTensor(g.node_features),
        edge_index=torch.LongTensor(g.edge_index),
        edge_attr=g.edge_features,
        num_nodes=g.num_nodes,
    )
    return torch_graph


def to_torch_graph_data_array_and_regr_y(mol_g, y):
    """Returns torch graph data from array of jaqpot graph_data and y"""
    import torch
    from torch_geometric.data import Data

    datas = []
    i = 0
    for g in mol_g:
        dato = Data(
            x=torch.FloatTensor(g.node_features),
            edge_index=torch.LongTensor(g.edge_index),
            edge_attr=g.edge_features,
            num_nodes=g.num_nodes,
            y=torch.Tensor([y[i]]),
        )
        datas.append(dato)
        i += 1
    return datas


def to_torch_graph_data_array_and_class_y(mol_g, y):
    """Returns torch graph data from array of jaqpot graph_data and y"""
    import torch
    from torch_geometric.data import Data

    datas = []
    i = 0
    for g in mol_g:
        dato = Data(
            x=torch.FloatTensor(g.node_features),
            edge_index=torch.LongTensor(g.edge_index),
            edge_attr=g.edge_features,
            num_nodes=g.num_nodes,
            y=torch.LongTensor([y[i]]),
        )
        datas.append(dato)
        i += 1
    return datas


def to_torch_graph_data_array(mol_g):
    """Returns torch graph data from array of jaqpot graph_data"""
    import torch
    from torch_geometric.data import Data

    datas = []
    for g in mol_g:
        dato = Data(
            x=torch.FloatTensor(g.node_features),
            edge_index=torch.LongTensor(g.edge_index),
            edge_attr=g.edge_features,
            num_nodes=g.num_nodes,
        )
        datas.append(dato)
    return datas
