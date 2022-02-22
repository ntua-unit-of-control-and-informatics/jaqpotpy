import pandas as pd
from jaqpotpy.datasets import SmilesDataset
from torch.utils.data import DataLoader
from jaqpotpy.descriptors import MordredDescriptors
import torch


smiles = ['O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
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


# df = pd.read_csv('./data/postera_model.csv')
#
# smiles = df['SMILES'].to_list()
# y = df['*f_avg_IC50'].to_list()
#
# len_act = 0
# ys = []
# for i in y:
#   if i < 90:
#       ys.append(1)
#       len_act += 1
#   else:
#       ys.append(0)


dataset = SmilesDataset(smiles=smiles, y=ys, featurizer=MordredDescriptors(ignore_3D=True),
                              task='classification')
# dataset.create()

dataloader = DataLoader(dataset, batch_size=40,
                shuffle=True, num_workers=0)


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
        # print(x)
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


model = Feedforward(1614, 3246)
# criterion = torch.nn.BCELoss()
criterion = torch.nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# if __name__ == '__main__':
def train():
    model.train()
    for data in dataloader:  # Iterate in batches over the training dataset.
        out = model(data[0].float())
        # print(out)
            # print(data[1].float())
        # out = torch.max(out, dim=1)
        target = data[1].float()
        loss = criterion(out.squeeze(), target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(loader):
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data[0].float())
            # pred = out.argmax(dim=1)
            # print(confusion_matrix(data.y, pred))
            # correct = int((pred == data.y).sum())
            # print(pred.numpy())
            # print(data[1].numpy())
        correct = int((out == data[1]).sum())
    return correct / len(loader.dataset)


for epoch in range(1, 40000):
    train()
    train_acc = test(dataloader)
    test_acc = test(dataloader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')