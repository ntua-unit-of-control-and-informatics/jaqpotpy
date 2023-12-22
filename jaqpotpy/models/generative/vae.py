from jaqpotpy.cfg import config
from jaqpotpy.datasets import SmilesDataset
from jaqpotpy.descriptors.molecular import MolGanFeaturizer, GraphMatrix
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers

import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import BondType
from rdkit.Chem.Draw import MolsToGridImage

RDLogger.DisableLog("rdApp.*")

csv_path = keras.utils.get_file(
    "./250k_rndm_zinc_drugs_clean_3.csv",
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
)

df = pd.read_csv("/Users/pantelispanka/.keras/datasets/250k_rndm_zinc_drugs_clean_3.csv")
df["smiles"] = df["smiles"].apply(lambda s: s.replace("\n", ""))
print(df.head())

smiles = df['smiles'].to_list()[0:1000]

feat = MolGanFeaturizer(max_atom_count=40)

dataset = SmilesDataset(smiles=smiles, featurizer=feat, task="generation")
dataset = dataset.create()
gm: GraphMatrix = dataset.df['MolGanGraphs'][0]


class Generator(torch.nn.Module):
    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0, norm: torch.nn.Module = None, act_first=True):
        super(Generator, self).__init__()
        torch.manual_seed(config.global_seed)

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

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


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0, norm: torch.nn.Module = None, act_first=True):
        super(Discriminator, self).__init__()
        torch.manual_seed(config.global_seed)
        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm
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
