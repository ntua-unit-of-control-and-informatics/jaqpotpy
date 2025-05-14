import torch
from torch_geometric.loader import DataLoader
import pandas as pd
from jaqpotpy import Jaqpot
from jaqpotpy.descriptors.graph import SmilesGraphFeaturizer
from jaqpotpy.datasets import SmilesGraphDataset
from jaqpotpy.models.torch_geometric_models.graph_neural_network import (
    GraphSageNetworkModel,
    pyg_to_onnx,
)
from jaqpotpy.models.trainers.graph_trainers import RegressionGraphModelTrainer

df = pd.read_csv("./jaqpotpy/test_data/test_data_smiles_regression.csv")

# Prepare Smiles and endpoint lists
# The endpoints can be either continuous or binary
train_smiles = list(df["SMILES"].iloc[:100])
train_y = list(df["ACTIVITY"].iloc[:100])

val_smiles = list(df["SMILES"].iloc[100:120])
val_y = list(df["ACTIVITY"].iloc[100:120])

test_smiles = list(df["SMILES"].iloc[120:150])
test_y = list(df["ACTIVITY"].iloc[120:150])

# Create an isntance of a graph featurizer
# Optional (Choose to include edgefeatures)
featurizer = SmilesGraphFeaturizer(include_edge_features=True)
# Obtain default node and edge feature values
featurizer.set_default_config()

# In case of creating custom node and edge attributes
# Node features and their can be hardcoded by user if it is supported by the featurizer
# Name of features and feature values should be provided for one hot encoding
# Calling featurizer.SUPPORTED_ATOM_FEATURES() provides the name of the features and their RDKit function to obtain them
featurizer.add_atom_feature("symbol", ["C", "O", "N", "F", "Cl", "Br", "I"])
featurizer.add_atom_feature("total_num_hs", [0, 1, 2, 3, 4])
# If feature value is not provided, one hot encoding will not be used
featurizer.add_atom_feature("formal_charge")

# With the same approach bond features can be added
# Calling featurizer.SUPPORTED_BOND_FEATURES() provides the name of the features and their RDKit function to obtain them

# Create datasets
# Specify smiles list, target list and created featurizer
train_dataset = SmilesGraphDataset(
    smiles=train_smiles, y=train_y, featurizer=featurizer
)
val_dataset = SmilesGraphDataset(smiles=val_smiles, y=val_y, featurizer=featurizer)

test_dataset = SmilesGraphDataset(smiles=test_smiles, y=test_y, featurizer=featurizer)

# In case of small datasets, precompute featurization is preffereable
# This will precompute the featurization of the dataset and store it in memory
train_dataset.precompute_featurization()
val_dataset.precompute_featurization()
test_dataset.precompute_featurization()


# Create a GraphNeuralNetwork architecture (GraphSageNetwork as an example)
# Obtain node_features from the featurizer
node_features = featurizer.get_num_node_features()
model = GraphSageNetworkModel(
    input_dim=node_features,  # Input neurons
    hidden_layers=1,  # Number of hidden layers
    hidden_dim=4,  # Hidden neurons
    output_dim=1,  # Output neurons (Default to 1)
    activation=torch.nn.ReLU(),  # Specify activation function from pytorch
    pooling="mean",  # Graph pooling (Readout function for graphs) (mean, add, max)
    dropout_proba=0.1,  # Dropout probability
)

# Specify pytorch optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = torch.nn.MSELoss()

# Create a trainer for Graph Regression Task
trainer = RegressionGraphModelTrainer(
    model=model, n_epochs=20, optimizer=optimizer, loss_fn=loss, scheduler=None
)  # Optionally scheduler can be provided from Pytorch

# Pytorch Geometric Data Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Train the model and evaluate on the validation set
trainer.train(train_loader, val_loader)

# Evaluate on the test set and obtain metrics
loss, metrics = trainer.evaluate(test_loader)

# Upload on jaqpot
# Convert pyg model to onnx for upload on Jaqpot
onnx_model = pyg_to_onnx(model, featurizer)
# Create an instance of Jaqpot
jaqpot = Jaqpot()
# Login to Jaqpot
jaqpot.login()
# Deploy the model on Jaqpot
jaqpot.deploy_torch_model(
    onnx_model,
    featurizer=featurizer,  # Featurizer used for the model
    name="Graph Sage Network",
    description="Graph Sage Network for regression",
    target_name="ACTIVITY",
    visibility="PRIVATE",
    task="regression",  # Specify the task (regression or binary_classification)
)
