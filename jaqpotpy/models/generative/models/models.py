import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import numpy as np
from jaqpotpy.models.generative.models.layers import GraphConvolution, GraphAggregation\
    , GraphConvolutionLayer, GraphConvolution2
from typing import Any, Iterable, Union, Dict


class GanGeneratorModel(object):
    _model: Any

    def sample(self, batch_size):
        raise NotImplemented("Not implemented")

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, value):
        self._descriptors = value

    @property
    def preprocessing(self):
        return self._preprocessors

    @preprocessing.setter
    def preprocessing(self, value):
        self._preprocessors = value

    @property
    def preprocessing_y(self):
        return self._preprocessors_y

    @preprocessing_y.setter
    def preprocessing_y(self, value):
        self._preprocessors_y = value

    @property
    def preprocessor_names(self):
        return self._preprocessor_names

    @preprocessor_names.setter
    def preprocessor_names(self, value):
        self._preprocessor_names = value

    @property
    def preprocessor_y_names(self):
        return self._preprocessor_names

    @preprocessor_y_names.setter
    def preprocessor_y_names(self, value):
        self._preprocessor_y_names = value

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, value):
        self._Y = value

    @property
    def external(self) -> dict:
        return self._external

    @external.setter
    def external(self, value):
        self._external = value

    @property
    def external_feats(self) -> Iterable[str]:
        return self._external_feats

    @external_feats.setter
    def external_feats(self, value):
        self._external_feats = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def model_title(self):
        return self._model_title

    @model_title.setter
    def model_title(self, value):
        self._model_title = value

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, value):
        self._prediction = value

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, value):
        self._probability = value

    @property
    def library(self):
        return self._library

    @library.setter
    def library(self, value):
        self._library = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, value):
        self._version = value

    @property
    def jaqpotpy_version(self):
        return self._jaqpotpy_version

    @jaqpotpy_version.setter
    def jaqpotpy_version(self, value):
        self._jaqpotpy_version = value

    @property
    def jaqpotpy_docker(self):
        return self._jaqpotpy_docker

    @jaqpotpy_docker.setter
    def jaqpotpy_docker(self, value):
        self._jaqpotpy_docker = value

    def __train__(self):
        raise NotImplemented("Not implemented")

    def __eval__(self):
        raise NotImplemented("Not implemented")

    def infer(self):
        raise NotImplemented("Not implemented")

class GanMoleculeGenerator(nn.Module, GanGeneratorModel):
    """Generator network."""
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout_rate):
        super(GanMoleculeGenerator, self).__init__()
        self.activation_f = torch.nn.Tanh()
        self.multi_dense_layer = MultiDenseLayer(z_dim, conv_dims, self.activation_f)

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)

        self.dropoout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.multi_dense_layer(x)
        edges_logits = self.edges_layer(output).view(-1, self.edges, self.vertexes, self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropoout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits

    def sample_generator(self, batch_size):
        return torch.Tensor(np.random.normal(0, 1, size=(batch_size, self.z_dim)))

    def sample(self, batch_size):
        return torch.Tensor(np.random.normal(0, 1, size=(batch_size, self.z_dim)))


class _GanMoleculeGenerator(nn.Module, GanGeneratorModel):
    """Generator network."""
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout):
        super(_GanMoleculeGenerator, self).__init__()

        self.z_dim = z_dim
        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        layers = []
        for c0, c1 in zip([z_dim]+conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout = nn.Dropout(p=dropout)

    def forward(self, x):
        output = self.layers(x)
        edges_logits = self.edges_layer(output)\
                       .view(-1, self.edges, self.vertexes, self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2))/2
        edges_logits = self.dropoout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1,self.vertexes,self.nodes))

        return edges_logits, nodes_logits

    def sample_generator(self, batch_size):
        return torch.Tensor(np.random.normal(0, 1, size=(batch_size, self.z_dim)))

    def sample(self, batch_size):
        return torch.Tensor(np.random.normal(0, 1, size=(batch_size, self.z_dim)))


class GanMoleculeDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, conv_dim, m_dim, b_dim, dropout):
        super(GanMoleculeDiscriminator, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GCNConv(m_dim, graph_conv_dim, b_dim, dropout)
        self.agg_layer = global_mean_pool(graph_conv_dim[-1], aux_dim, b_dim, dropout)

        # multi dense layer
        layers = []
        for c0, c1 in zip([aux_dim]+linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, hidden, node, activatation=None):
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        annotations = torch.cat((hidden, node), -1) if hidden is not None else node
        h = self.gcn_layer(annotations, adj)
        annotations = torch.cat((h, hidden, node) if hidden is not None\
                                 else (h, node), -1)
        h = self.agg_layer(annotations, torch.tanh)
        h = self.linear_layer(h)

        # Need to implemente batch discriminator #
        ##########################################

        output = self.output_layer(h)
        output = activatation(output) if activatation is not None else output

        return output, h


class Discriminator2(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim, m_dim, b_dim, with_features=False, f_dim=0, dropout_rate=0.):
        super(Discriminator2, self).__init__()
        self.activation_f = torch.nn.Tanh()
        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution2(m_dim, graph_conv_dim, b_dim, with_features, f_dim, dropout_rate)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1] + m_dim, aux_dim, self.activation_f, with_features, f_dim,
                                          dropout_rate)
        self.multi_dense_layer = MultiDenseLayer(aux_dim, linear_dim, self.activation_f, dropout_rate=dropout_rate)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, hidden, node, activation=None):
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        h_1 = self.gcn_layer(node, adj, hidden)
        h = self.agg_layer(h_1, node, hidden)
        h = self.multi_dense_layer(h)

        output = self.output_layer(h)
        output = activation(output) if activation is not None else output

        return output, h


class MoleculeDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim, m_dim, b_dim, with_features=False, f_dim=0, dropout_rate=0.):
        super(MoleculeDiscriminator, self).__init__()
        self.activation_f = torch.nn.Tanh()
        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, with_features, f_dim, dropout_rate)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1] + m_dim, aux_dim, self.activation_f, with_features, f_dim,
                                          dropout_rate)
        self.multi_dense_layer = MultiDenseLayer(aux_dim, linear_dim, self.activation_f, dropout_rate=dropout_rate)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, hidden, node, activation=None):
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        h_1 = self.gcn_layer(node, adj, hidden)
        h = self.agg_layer(h_1, node, hidden)
        h = self.multi_dense_layer(h)
        output = self.output_layer(h)
        output = activation(output) if activation is not None else output
        return output, h


class MultiDenseLayer(nn.Module):
    def __init__(self, aux_unit, linear_units, activation=None, dropout_rate=0.):
        super(MultiDenseLayer, self).__init__()
        layers = []
        for c0, c1 in zip([aux_unit] + linear_units[:-1], linear_units):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Dropout(dropout_rate))
            if activation is not None:
                layers.append(activation)
        self.linear_layer = nn.Sequential(*layers)

    def forward(self, inputs):
        h = self.linear_layer(inputs)
        return h