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

ys = []
for i in y:
  if i < 100:
      ys.append(1)
  else:
      ys.append(0)


mol_graph = MolGraphConvFeaturizer(use_edges=True)
mol_g_desc = mol_graph.featurize(smiles)
graphs = ptu.to_torch_graph_data_array_and_class_y(mol_g_desc, ys)

print(graphs[0])
print(graphs[0].__dict__)
# for i in graphs:
#     print(type(i))

# mol_graph = TorchMolGraphConvFeaturizer(use_edges=True)
# mol_g_desc = mol_graph.featurize(smiles)
#
# graphs = []
# i =0
# for g in mol_g_desc:
#     # print(g.shape)
#     # print(type(g[0][1]))
#     # g.insert(['y': y[i]])
#     g['y'] = y[i]
#     i += 1
#     graphs.append(g)


# i=0
# datas = []
# for g in mol_g_desc:
#     if type(y[i]) is not None:
#         dato = Data(x=torch.FloatTensor(g.node_features)
#                     , edge_index=torch.LongTensor(g.edge_index)
#                     , edge_attr=g.edge_features
#                     , num_nodes=g.num_nodes, y=np.array([y[i]]))
#         # dato = Data(x=torch.LongTensor(g.node_features)
#         #             , edge_index=torch.LongTensor(g.edge_index)
#         #             , edge_attr=torch.LongTensor(g.edge_features)
#         #             , num_nodes=g.num_nodes, y=torch.FloatTensor(np.array([y[i]])))
#         datas.append(dato)
#         i += 1
# print(datas)
# dataset = DataLoader(datas, batch_size=64, shuffle=True)


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(30, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(16, 2)

        # super(GCN, self).__init__()
        # torch.manual_seed(12345)
        # self.conv1 = GCNConv(30, 22)
        # self.conv2 = GCNConv(22, 22)
        # self.conv3 = GCNConv(22, 22)
        # self.lin = Linear(22, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x
        # x, edge_index = data.x, data.edge_index
        # x = self.conv1(x, edge_index)
        # x = x.relu()
        # x = self.conv2(x, edge_index)
        # # x = x.relu()
        # # x = self.conv3(x, edge_index)
        #
        # # 2. Readout layer
        # x = global_mean_pool(x, data.batch)  # [batch_size, hidden_channels]
        #
        # # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin(x)
        #
        # return x


train_dataset = graphs[39:]
test_dataset = graphs[:39]
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN() #.to(device)
# dataset = DataLoader(train_dataset, batch_size=1, shuffle=True) #.to(device)
train_loader = DataLoader(train_dataset, batch_size=44, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=39, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.MSELoss()
# criterion = torch.nn.L1Loss()
criterion = torch.nn.CrossEntropyLoss()

# def train():
#     model.train()
#     for data in train_loader:  # Iterate in batches over the training dataset.
#         out = model(data)
#         loss = criterion(out, torch.FloatTensor(data.y))
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#
# def test(loader):
#     model.eval()
#     for data in loader:  # Iterate in batches over the training/test dataset.
#         out = model(data)
#         loss = criterion(out, torch.FloatTensor(data.y))
#     return loss


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
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data)
        pred = out.argmax(dim=1)
        # correct = int((pred == data.y).sum())
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


for epoch in range(1, 40000):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# model.train()
# for epoch in range(10):
#     for data in train_loader:
#         # optimizer.zero_grad()
#         out = model(data)
#         loss = criterion(out, data.y)
#         optimizer.step()
#         optimizer.zero_grad()



