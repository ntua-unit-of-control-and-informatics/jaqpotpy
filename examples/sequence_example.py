import pandas as pd
from rdkit import Chem
import torch
import torch.nn as nn
from jaqpotpy.models.torch_models.smiles_sequence import Sequence_LSTM, lstm_to_onnx
from jaqpotpy.descriptors.tokenizer import SmilesVectorizer
from jaqpotpy.datasets.tokenizer_dataset import SmilesSeqDataset
from jaqpotpy.models.trainers.sequence_trainers import BinarySequenceTrainer
from torch.utils.data import DataLoader
import onnxruntime
import base64

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

trainer = BinarySequenceTrainer(
    model,
    n_epochs=2,
    optimizer=optimizer,
    loss_fn=criterion,
    device=device,
)
trainer.train(train_loader, test_loader)
example = test_dataset.X[0:2]
torch.manual_seed(0)
torch_pred = model(example)
print(torch_pred)
onnx_model = lstm_to_onnx(model, train_dataset)


#### Will be used for inference
onnx_model = base64.b64decode(onnx_model)


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


ort_session = onnxruntime.InferenceSession(onnx_model)
# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(example)}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)
