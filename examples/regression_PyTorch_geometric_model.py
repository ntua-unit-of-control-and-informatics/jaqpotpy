import pandas as pd
import torch as t
from jaqpotpy import Jaqpot
from jaqpotpy.descriptors.graph import SmilesGraphFeaturizer
from jaqpotpy.datasets import SmilesGraphDataset
from jaqpotpy.models.torch_geometric_models.graph_neural_network import (
    GraphSageNetwork,
    pyg_to_onnx,
)
from jaqpotpy.models.trainers.graph_trainers import RegressionGraphModelTrainer
from torch_geometric.loader import DataLoader

df = pd.read_csv("./jaqpotpy/test_data/test_data_smiles_regression.csv")

# Prepare Smiles and endpoint lists
train_smiles = list(df["SMILES"].iloc[:100])
train_y = list(df["ACTIVITY"].iloc[:100])

val_smiles = list(df["SMILES"].iloc[100:120])
val_y = list(df["ACTIVITY"].iloc[100:120])

test_smiles = list(df["SMILES"].iloc[120:150])
test_y = list(df["ACTIVITY"].iloc[120:150])

### Featurizer
featurizer = SmilesGraphFeaturizer()
# Add a node feature
featurizer.add_atom_feature("symbol", ["C", "O", "N", "F", "Cl", "Br", "I"])

### Dataset

train_dataset = SmilesGraphDataset(
    smiles=train_smiles, y=train_y, featurizer=featurizer
)
val_dataset = SmilesGraphDataset(smiles=val_smiles, y=val_y, featurizer=featurizer)

test_dataset = SmilesGraphDataset(smiles=test_smiles, y=test_y, featurizer=featurizer)

train_dataset.precompute_featurization()
val_dataset.precompute_featurization()
test_dataset.precompute_featurization()


## Model

node_features = featurizer.get_num_node_features()
model = GraphSageNetwork(
    input_dim=node_features,
    hidden_layers=1,
    hidden_dim=4,
    output_dim=1,
    activation=t.nn.ReLU(),
    pooling="mean",
    dropout_proba=0.1,
)

# Pytorch args
optimizer = t.optim.Adam(model.parameters(), lr=0.001)
loss = t.nn.MSELoss()
epochs = 5

# Trainer
trainer = RegressionGraphModelTrainer(model, epochs, optimizer, loss)
# Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
# train and evaluate
trainer.train(train_loader, val_loader)
# Evaluate on test set
loss, metrics = trainer.evaluate(test_loader)
print(metrics)

## Upload on jaqpot

# Convert pyg model to onnx
onnx_model = pyg_to_onnx(model, featurizer)

jaqpot = Jaqpot()
jaqpot.login()

jaqpot.deploy_torch_model(
    onnx_model,
    featurizer=featurizer,
    name="Graph Sage Network",
    description="Graph Sage Network for regression",
    target_name="ACTIVITY",
    visibility="PRIVATE",
    task="regression",
)
