import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from jaqpotpy.models.evaluator import Evaluator
from jaqpotpy.models import MolecularTorchGeometric
from jaqpotpy.datasets import TorchGraphDataset


from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, jaccard_score, confusion_matrix


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(30, 40)
        self.conv2 = GCNConv(40, 40)
        self.conv3 = GCNConv(40, 40)
        self.lin = Linear(40, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


mols = ['O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
    , 'O=C1CCc2cc(C(=O)N3CCC(C4CCNC4)CC3)ccc2N1'
    , 'CCC(=O)Nc1ccc(N(Cc2ccccc2)C(=O)n2nnc3ccccc32)cc1'
    , 'COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'
    , 'Cc1nn(C)c2[nH]nc(NC(=O)Cc3cccc(Cl)c3)c12'
    , 'O=C(Cc1cncc2ccccc12)N(CCC1CCCCC1)c1cccc(Cl)c1'
    , 'COc1ccc(N(Cc2ccccc2)C(=O)Cc2c[nH]c3ccccc23)cc1'
    , 'CC(C)(C)c1ccc(N(C(=O)c2ccco2)[C@H](C(=O)NCCc2cccc(F)c2)c2cccnc2)cc1'
    , 'OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2'
    , 'Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2cnn(C)c2C(F)F)c1'
    , 'Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)Cc1cccc(Cl)c1'
    , 'Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)C1CCOc2ccc(Cl)cc21'
    , 'O=C(c1cc(=O)[nH]c2ccccc12)N1CCN(c2cccc(Cl)c2)C(=O)C1'
    , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
    , 'COCCNC(=O)[C@@H](c1ccccc1)N1Cc2ccccc2C1=O'
    , 'CNCC1CCCN(C(=O)[C@@H](c2ccccc2)N2Cc3ccccc3C2=O)C1'
    , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
    , 'COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1'
    , 'O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21'
    , 'COc1ccccc1OC1CCN(C(=O)c2cc(=O)[nH]c3ccccc23)C1'
    , 'O=C(Cc1cc(Cl)cc(Cc2ccn[nH]2)c1)Nc1cncc2ccccc12'
    , 'CN(C)c1ccc(N(Cc2ccsc2)C(=O)Cc2cncc3ccccc23)cc1'
    , 'C[C@H]1COc2ccc(Cl)cc2[C@@H]1C(=O)Nc1cncc2ccccc12'
        ]

ys = [
    0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1
]

dataset = TorchGraphDataset(smiles=mols, y=ys, task='classification')
dataset.create()
val = Evaluator()
val.dataset = dataset
val.register_scoring_function('Accuracy', accuracy_score)
val.register_scoring_function('f1', f1_score)
val.register_scoring_function('Roc AUC', roc_auc_score)
val.register_scoring_function('Jaccard', jaccard_score)
val.register_scoring_function("Confusion Matrix", confusion_matrix)
model_nn = GCN()
optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
mlm = MolecularTorchGeometric( dataset=dataset
                              , model_nn=model_nn, eval=val
                              , train_batch=12, test_batch=10
                              , epochs=400, optimizer=optimizer, criterion=criterion, log_steps=50).fit()
mlm.eval()

print("GOOIING REGRESSION")


class GCN2(torch.nn.Module):
    def __init__(self):
        super(GCN2, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(30, 40)
        self.conv2 = GCNConv(40, 40)
        self.conv3 = GCNConv(40, 40)
        self.lin = Linear(40, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


dataset2 = TorchGraphDataset(smiles=mols, y=ys, task='regression')
dataset2.create()
val2 = Evaluator()
val2.dataset = dataset2
val2.register_scoring_function('Max Error', max_error)
val2.register_scoring_function('Mean Absolute Error', mean_absolute_error)
val2.register_scoring_function('R 2 score', r2_score)
model_nn2 = GCN2()
optimizer = torch.optim.Adam(model_nn2.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.L1Loss()
mlm2 = MolecularTorchGeometric(dataset=dataset2
                              , model_nn=model_nn2, eval=val2
                              , train_batch=12, test_batch=10
                              , epochs=400, optimizer=optimizer, criterion=criterion, log_steps=50)
mlm2.fit()
mlm2.eval()


print("GOING SIMPLE")


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

