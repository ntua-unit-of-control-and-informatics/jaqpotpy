import pandas as pd

df = pd.read_csv("jaqpotpy/test_data/test_data_smiles_classification.csv")

train_smiles = list(df["SMILES"].iloc[:100])
train_y = list(df["ACTIVITY"].iloc[:100])

val_smiles = list(df["SMILES"].iloc[100:200])
val_y = list(df["ACTIVITY"].iloc[100:200])

from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer
from rdkit import Chem

featurizer = SmilesGraphFeaturizer()
featurizer.add_atom_feature("symbol", ["C", "O", "N", "F", "Cl", "Br", "I"])

featurizer.add_bond_feature(
    "bond_type",
    [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
)

from jaqpotpy.datasets import SmilesGraphDataset

train_dataset = SmilesGraphDataset(
    smiles=train_smiles, y=train_y, featurizer=featurizer
)
val_dataset = SmilesGraphDataset(smiles=val_smiles, y=val_y, featurizer=featurizer)

train_dataset.precompute_featurization()
val_dataset.precompute_featurization()

from jaqpotpy.models.torch_geometric_models.graph_neural_network import (
    GraphAttentionNetwork,
    pyg_to_torchscript,
)

input_dim = featurizer.get_num_node_features()
edge_dim = featurizer.get_num_edge_features()

model = GraphAttentionNetwork(
    input_dim=input_dim,
    hidden_layers=2,
    hidden_dim=16,
    output_dim=1,
    dropout_proba=0.5,
    graph_norm=False,  # Doesnt work with graph_norm
    seed=42,
    edge_dim=edge_dim,
    heads=4,
)

from jaqpotpy.models.trainers import BinaryGraphModelTrainer
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

from torch_geometric.loader import DataLoader

optimizer = Adam(model.parameters(), lr=0.001)
loss = BCEWithLogitsLoss()
epochs = 5
trainer = BinaryGraphModelTrainer(model, epochs, optimizer, loss)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
trainer.train(train_loader, val_loader)

torchscript_model = pyg_to_torchscript(model)
from jaqpotpy import Jaqpot

jaqpot = Jaqpot()
jaqpot.login()
# jaqpot.set_api_key(
#     "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJEQTQtalctUDRzRzc2cHM4WlNFUHVzZTdxQWNTUVUtSjFCcURjR0g1NXFRIn0.eyJleHAiOjE3MjczNjk2ODQsImlhdCI6MTcyNzMzMzY4NCwianRpIjoiY2ZiYzcyOGEtNDAyZC00NWU4LWE0MmQtMjc4YTAzOGIyNjg5IiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdC5qYXFwb3Qub3JnOjgwNzAvcmVhbG1zL2phcXBvdC1sb2NhbCIsImF1ZCI6ImFjY291bnQiLCJzdWIiOiIyNjIwYTAxZi1kZWIxLTQ4MGQtOWYzMC1hNmI0MTI4YWFkMWEiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJqYXFwb3QtbG9jYWwtdGVzdCIsInNlc3Npb25fc3RhdGUiOiJlZTYyMjYxMS1kMmMyLTRiZWQtOWZjYS0xMDZlNWUwMjAzYTgiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIiIsIioiXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbImRlZmF1bHQtcm9sZXMtamFxcG90LWxvY2FsIiwib2ZmbGluZV9hY2Nlc3MiLCJhZG1pbiIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJlbWFpbCBwcm9maWxlIiwic2lkIjoiZWU2MjI2MTEtZDJjMi00YmVkLTlmY2EtMTA2ZTVlMDIwM2E4IiwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5hbWUiOiJqYXFwb3QgYWRtaW4iLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqYXFwb3QtYWRtaW4iLCJnaXZlbl9uYW1lIjoiamFxcG90IiwiZmFtaWx5X25hbWUiOiJhZG1pbiIsImVtYWlsIjoiamFxcG90LWFkbWluQGphcXBvdC5vcmcifQ.eX5viBvNEVxWz0BLOQojwdlgHl9eQri3X2hXHF6CE9ifwvwyZStTatO-ZGoVHzDneftMaRZcifOMugIg8tEmdGbBR8K_SSSKGBPyed9MjR21HAGEGkhhHrD9Mzu5peQaQ-LvCxsvHZBHyZwzofyw2-6fuGKfHgkFxJq8o_TELniE4UhtBDj4KA7dyMZu1pBw3yH0bbRwdYfq7JrOAsRbhRlxbMK_U_fSm4ShgwcHI9oLTRlVn84YxVq-e6LfbUrZ2BHDHXJzCtGL50tCzPXoXfC8LlQybrlCy6JgfpxRcYjbXHndzm1tNbVzowpSyRdsBxA3BaH7-Q3r51_PGrPf4A"
# )

jaqpot.deploy_torch_model(
    torchscript_model,
    type="TORCHSCRIPT",
    featurizer=featurizer,
    name="Graph Attention Network with torchscript",
    description="Graph Attention Network for binary classification",
    target_name="ACTIVITY",
    visibility="PRIVATE",
    task="binary_classification",
)
