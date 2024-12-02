import torch as t
from rdkit import Chem
from jaqpotpy import Jaqpot
from jaqpotpy.models.torch_models import Sequence_LSTM
from jaqpotpy.models.trainers.sequence_trainers import RegressionSequenceTrainer
from jaqpotpy.datasets import SmilesSeqDataset
from jaqpotpy.descriptors.tokenizer import SmilesVectorizer
from tdc.single_pred import ADME

data = ADME(name="Lipophilicity_AstraZeneca")
split = data.get_split()

train_smiles = split["train"]["Drug"]
train_y = split["train"]["Y"]
val_smiles = split["valid"]["Drug"]
val_y = split["valid"]["Y"]
test_smiles = split["test"]["Drug"]
test_y = split["test"]["Y"]

tokenizer = SmilesVectorizer()
# train_mols = [Chem.MolFromSmiles(smile) for smile in train_smiles]
# val_mols = [Chem.MolFromSmiles(smile) for smile in val_smiles]
# test_mols = [Chem.MolFromSmiles(smile) for smile in test_smiles]
tokenizer.fit(train_smiles)
train_dataset = SmilesSeqDataset(train_smiles, train_y, tokenizer)
val_dataset = SmilesSeqDataset(val_smiles, val_y, tokenizer)
test_dataset = SmilesSeqDataset(test_smiles, test_y, tokenizer)

train_loader = t.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = t.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = t.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

model = Sequence_LSTM(
    input_size=tokenizer.dims[1],
    hidden_size=32,
    num_layers=1,
    output_size=1,
    dropout=0.2,
    activation=t.nn.ReLU(),
)

loss = t.nn.MSELoss()
optimizer = t.optim.Adam(model.parameters(), lr=0.001)
trainer = RegressionSequenceTrainer(
    model=model, n_epochs=10, optimizer=optimizer, loss_fn=loss
)

trainer.train(train_loader=train_loader, val_loader=val_loader)

_, val_metrics = trainer.evaluate(val_loader)
print(val_metrics)

_, test_metrics = trainer.evaluate(test_loader)
print(test_metrics)
