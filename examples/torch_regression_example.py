import pandas as pd
from jaqpotpy.descriptors import MolGraphConvFeaturizer, TorchMolGraphConvFeaturizer
import jaqpotpy.utils.pytorch_utils as ptu
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import TUDataset, MoleculeNet
import numpy as np

df = pd.read_csv('/Users/pantelispanka/Downloads/ecoli_DNA_gyrase_subunit_B_reductase_ic50.csv')

smiles = df['canonical_smiles'].to_list()
y = df['standard_value'].to_list()


mol_graph = MolGraphConvFeaturizer(use_edges=True)
mol_g_desc = mol_graph.featurize(smiles)
graphs = ptu.to_torch_graph_data_array_and_regr_y(mol_g_desc, y)


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(30, 42)
        self.conv2 = GCNConv(42, 42)
        self.lin = Linear(42, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.lin(x)
        return x


train_dataset = graphs[:110]
test_dataset = graphs[110:]
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN() #.to(device)
# dataset = DataLoader(train_dataset, batch_size=1, shuffle=True) #.to(device)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=29, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# criterion = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()


def train():
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(loader):
    model.eval()
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data)
        loss = criterion(out, data.y)
    return loss


for epoch in range(1, 40000):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train MSELoss: {train_acc:.4f}, Test err: {test_acc:.4f}')

# model.train()
# for epoch in range(10):
#     for data in train_loader:
#         # optimizer.zero_grad()
#         out = model(data)
#         loss = criterion(out, data.y)
#         optimizer.step()
#         optimizer.zero_grad()
