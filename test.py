import pandas as pd
from rdkit import Chem
import torch
import torch.nn as nn
from jaqpotpy.models.torch_models.smiles_sequence import Sequence_LSTM
from jaqpotpy.descriptors.tokenizer import SmilesVectorizer
from jaqpotpy.datasets.tokenizer_dataset import SmilesSeqDataset
from torch.utils.data import DataLoader

path = "AllPublicnew.csv"
smiles_train = list(pd.read_csv(path).iloc[0:500, :]["SMILES"])
y_train = list(pd.read_csv(path).iloc[0:500, :]["ReadyBiodegradability"])
smiles_test = list(pd.read_csv(path).iloc[600:700, :]["SMILES"])
y_test = list(pd.read_csv(path).iloc[600:700, :]["ReadyBiodegradability"])

mols1 = [Chem.MolFromSmiles(smile) for smile in smiles_train]
mols2 = [Chem.MolFromSmiles(smile) for smile in smiles_test]

smivec = SmilesVectorizer(pad=1, leftpad=True, canonical=False, augment=False)
train_dataset = SmilesSeqDataset(mols1, y_train, smivec)
test_dataset = SmilesSeqDataset(mols2, y_test, smivec)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Sequence_LSTM(
    input_size=train_dataset.get_feature_dim(),
    hidden_size=128,
    num_layers=1,
    dropout=0.5,
    output_size=1,
    activation=nn.ReLU(),
    bidirectional=False,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
criterion = nn.BCEWithLogitsLoss()

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(dim=-1).float())
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
    running_loss = 0
    for inputs, labels in test_loader:
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(dim=-1).float())
            running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(test_loader)}")
    #
    # print(
    #     f"Accuracy: {((outputs > 0).float() == labels.unsqueeze(dim=-1)).float().mean()}"
    # )
