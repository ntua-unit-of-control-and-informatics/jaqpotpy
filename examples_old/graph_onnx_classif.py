import pandas as pd
from jaqpotpy.descriptors.graph import SmilesGraphFeaturizer
from rdkit import Chem
from jaqpotpy.datasets import SmilesGraphDataset
from jaqpotpy.models.torch_geometric_models.graph_neural_network import (
    GraphSageNetwork,
    pyg_to_onnx,
)
from jaqpotpy.models.trainers.graph_trainers import BinaryGraphModelTrainer
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

from torch_geometric.loader import DataLoader
from jaqpotpy import Jaqpot

df = pd.read_csv("./jaqpotpy/test_data/test_data_smiles_classification.csv")

train_smiles = list(df["SMILES"].iloc[:100])
train_y = list(df["ACTIVITY"].iloc[:100])

val_smiles = list(df["SMILES"].iloc[100:200])
val_y = list(df["ACTIVITY"].iloc[100:200])

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

train_dataset = SmilesGraphDataset(
    smiles=train_smiles, y=train_y, featurizer=featurizer
)
val_dataset = SmilesGraphDataset(smiles=val_smiles, y=val_y, featurizer=featurizer)

train_dataset.precompute_featurization()
val_dataset.precompute_featurization()

input_dim = featurizer.get_num_node_features()
edge_dim = featurizer.get_num_edge_features()

model = GraphSageNetwork(
    input_dim=input_dim,
    hidden_layers=2,
    hidden_dim=16,
    output_dim=1,
    dropout_proba=0.5,
    seed=42,
)

optimizer = Adam(model.parameters(), lr=0.001)
loss = BCEWithLogitsLoss()
epochs = 5
trainer = BinaryGraphModelTrainer(model, epochs, optimizer, loss)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
trainer.train(train_loader, val_loader)

onnx_model = pyg_to_onnx(model, featurizer)

jaqpot = Jaqpot()
jaqpot.login()

jaqpot.deploy_torch_model(
    onnx_model,
    featurizer=featurizer,
    name="Graph Sage Network",
    description="Graph Sage Network for binary classification",
    target_name="ACTIVITY",
    visibility="PRIVATE",
    task="binary_classification",
)
