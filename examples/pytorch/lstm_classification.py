import torch as t
import pandas as pd
import numpy as np
import base64
from rdkit import Chem
from jaqpotpy import Jaqpot
from jaqpotpy.models.torch_models import SequenceLstmModel, lstm_to_onnx
from jaqpotpy.models.trainers.sequence_trainers import BinarySequenceTrainer
from jaqpotpy.datasets import SmilesSeqDataset
from jaqpotpy.descriptors.tokenizer import SmilesVectorizer
from jaqpot_api_client.models.model_task import ModelTask
from jaqpot_api_client.models.model_visibility import ModelVisibility
from tdc.single_pred import Tox
import onnxruntime
import random

np.random.seed(42)
t.manual_seed(42)
random.seed(42)

data = Tox(name="AMES")
split = data.get_split()

train_smiles = split["train"]["Drug"][:1000]
train_y = split["train"]["Y"][:1000]
val_smiles = split["valid"]["Drug"][:1000]
val_y = split["valid"]["Y"][:1000]
test_smiles = split["test"]["Drug"][:1000]
test_y = split["test"]["Y"][:1000]

tokenizer = SmilesVectorizer()
tokenizer.fit(train_smiles)

train_dataset = SmilesSeqDataset(train_smiles, train_y, tokenizer)
val_dataset = SmilesSeqDataset(val_smiles, val_y, tokenizer)
test_dataset = SmilesSeqDataset(test_smiles, test_y, tokenizer)
train_loader = t.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = t.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = t.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

model = SequenceLstmModel(
    input_size=tokenizer.dims[1],
    hidden_size=64,
    num_layers=1,
    output_size=1,
    dropout=0.2,
    activation=t.nn.ReLU(),
)

loss = t.nn.BCEWithLogitsLoss()
optimizer = t.optim.Adam(model.parameters(), lr=0.001)
trainer = BinarySequenceTrainer(
    model=model, n_epochs=10, optimizer=optimizer, loss_fn=loss
)

trainer.train(train_loader=train_loader, val_loader=val_loader)

_, val_metrics, _ = trainer.evaluate(val_loader)
print(val_metrics)

_, test_metrics, _ = trainer.evaluate(test_loader)
print(test_metrics)

onnx_model = onnx_model = lstm_to_onnx(model, tokenizer)
# # Login to Jaqpot
jaqpot = Jaqpot()
jaqpot.login()
# Deploy the model on Jaqpot
jaqpot.deploy_torch_model(
    onnx_model,
    featurizer=tokenizer,
    name="LSTM",
    description="LSTM for binary classification",
    target_name="ACTIVITY",
    visibility=ModelVisibility.PRIVATE,
    task=ModelTask.BINARY_CLASSIFICATION,  # Specify the task (regression or binary_classification)
)
