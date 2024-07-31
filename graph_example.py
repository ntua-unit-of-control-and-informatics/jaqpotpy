smiles_train = [
    'COc1cccc(S(=O)(=O)Cc2cc(C(=O)NO)no2)c1',
    'O=C(N[C@H](CO)[C@H](O)c1ccc([N+](=O)[O-])cc1)C(Cl)Cl',
    'CC(=O)NC[C@H]1CN(c2ccc3c(c2F)CCCCC3=O)C(=O)O1',
    'CC(C)(C)OC(=O)C=C',
    'CC1(CC(CC(C1)(C)CN=C=O)N=C=O)C',
    'C1=CN=CN1'
]

smiles_val = [
    'CC(=O)NC[C@H]1CN(c2ccc(C(C)=O)cc2)C(=O)O1',
    'NCCNC(=O)COc1c2cccc1Cc1cccc(c1O)Cc1cccc(c1OCC(=O)NCCN)Cc1cccc(c1O)C2',
    'C1=CC=CC=C1',
    'CCCCCCC',
    'OP(=O)(O)[O-].[Na+]'
]


# Binary labels
y_train = [0, 1, 1, 0, 1, 1]
y_val = [1, 0, 0, 0, 1]

from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer
from rdkit import Chem

featurizer = SmilesGraphFeaturizer()

featurizer.add_atom_feature('symbol', ['C', 'O', 'N', 'Cl', 'S', 'F', 'Na', 'P', 'Br', 'Si', 'K', 'Sn', 'UNK'])
featurizer.add_atom_feature('degree', [0, 1, 2, 3, 4])
featurizer.add_atom_feature('hybridization', [Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3,
                        Chem.rdchem.HybridizationType.S,
                        Chem.rdchem.HybridizationType.SP])
featurizer.add_atom_feature('implicit_valence', [0, 1, 2, 3, 4])
featurizer.add_atom_feature('is_aromatic')

#json = featurizer.get_json_rep()
#print(type(json))

#feat = SmilesGraphFeaturizer()
#feat.load_json_rep(json)
#print(feat)
from jaqpotpy.datasets.graph_pyg_dataset import SmilesGraphDataset

train_dataset = SmilesGraphDataset(smiles_train, y_train, featurizer=featurizer)
val_dataset = SmilesGraphDataset(smiles_val, y_val, featurizer=featurizer)

print(f"Train dataset consists of {len(train_dataset)} samples.")
print(f"Val dataset consists of {len(val_dataset)} samples.")

train_dataset.precompute_featurization()
val_dataset.precompute_featurization()

from jaqpotpy.models.torch_geometric_models.graph_convolutional_network import GraphConvolutionalNetwork

input_dim = featurizer.get_num_node_features()
edge_dim = featurizer.get_num_edge_features()
# edge_dim = None

model = GraphConvolutionalNetwork(input_dim=input_dim,
                              hidden_dims=[5, 5],
                              output_dim=1,
                              dropout=0.5,
                              graph_norm=False)

import torch
import torch.nn as nn

n_epochs = 3
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.BCEWithLogitsLoss()

# DEVICE
seed = 42
use_gpu = True # Change this for use of CPU or GPU (if available)
torch.manual_seed(seed)
if torch.cuda.is_available() and use_gpu:
    device = torch.device('cuda:0')
    torch.cuda.manual_seed(seed)
    print(f"\nDevice: \n- {torch.cuda.get_device_name(device=device)}")
else:
    device = torch.device('cpu')
    print("\nDevice: 'CPU'")

from jaqpotpy.models.trainers.binary_trainers import BinaryGraphModelTrainer

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
trainer = BinaryGraphModelTrainer(model,
                                  n_epochs=n_epochs,
                                  optimizer=optimizer,
                                  loss_fn=loss_fn,
                                  device=device,
                                  log_enabled=True)

trainer.train(train_loader, val_loader=val_loader)

json = trainer.prepare_for_deployment(featurizer=featurizer,
                               endpoint_name='test',
                               name='SMILES-Binary-Classifier')

from jaqpotpy.jaqpot import Jaqpot
jaqpot = Jaqpot(base_url='http://localhost.jaqpot.org:8080/')

jaqpot.set_api_key("eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI5LVBxT3loU19xTmYtaDFzNmQ5SXZzN29PMDM5dmEzeElKeURsOTZpd2ZRIn0.eyJleHAiOjE3MjI0NjM1ODksImlhdCI6MTcyMjQyNzU4OSwianRpIjoiMTQ1ZWMzM2YtODlkMi00YTczLTkxNzQtZjliZjNlYzRhZjlhIiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdC5qYXFwb3Qub3JnOjgwNzAvcmVhbG1zL2phcXBvdC1sb2NhbCIsImF1ZCI6ImFjY291bnQiLCJzdWIiOiJiZWM0NDkzNS04ZjU2LTRiOTgtYThiYi0wZTdlNWUwYjA5N2EiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJqYXFwb3QtbG9jYWwtdGVzdCIsInNlc3Npb25fc3RhdGUiOiIyY2M4NTdkNy02NzA1LTRjNmQtYTY5My1iZmZmNTg2YzA2YzEiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIiIsIioiXSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoiZW1haWwgcHJvZmlsZSIsInNpZCI6IjJjYzg1N2Q3LTY3MDUtNGM2ZC1hNjkzLWJmZmY1ODZjMDZjMSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYW1lIjoiamFxIHBvdCIsInByZWZlcnJlZF91c2VybmFtZSI6ImphcXBvdCIsImdpdmVuX25hbWUiOiJqYXEiLCJmYW1pbHlfbmFtZSI6InBvdCIsImVtYWlsIjoiamFxcG90QGphcXBvdC5vcmcifQ.p_iwZ7QlGT90KM4IfTyY0U2ZmEom9GqGbScr9LbD5W94F137Oay5MhHtOhNfKe3PJUCtMLZZzVSWPgp_IVZBhTRMVIVE09epotbM6Tn_pfOSSDHM2HyIYmiUXd3mATRSviVUMw8AZ9T-MgaD4n4weY2q8b-qF6gblEIKXGkkIW44R9lfEJa6JJ3WEjSR26aXHgnUklnnmzub7J4KkOKvyajVKV8QcuhycOziTHb2DpOhzZEkS1RP33jRlQYH74k3dN3ebXjEyd4ZaJgIpJMsruGLopihohlXfJKultK7bAZbN1RzDRF33BAR6GYuk9tp-PY50RlgWQyPhFXx9Of9VQ")
jaqpot.deploy_Torch_Graph_model(json)


###
#print(json.get('actualModel'))

#print(json)

#### Dummy inference


# import onnxruntime
# import base64
# import io

# #print(json['actualModel'])
# onnx_model = json['actualModel']
# #onxx_model = base64.b64decode(onnx_model)
# #onnx_model = io.BytesIO(onnx_model)
# #ort_session = onnxruntime.InferenceSession(onnx_model)
# ort_session = onnxruntime.InferenceSession(onnx_model)  

# smile = 'CC(=O)NC[C@H]1CN(c2ccc(C(C)=O)cc2)C(=O)O1'
# feat = json['additional_model_params']['featurizer']

# featurizer = SmilesGraphFeaturizer()
# featurizer.load_json_rep(feat)
# data = featurizer(smile)

# def to_numpy(tensor):
#      return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data.x),
#               ort_session.get_inputs()[1].name: to_numpy(data.edge_index),
#               ort_session.get_inputs()[2].name: to_numpy(torch.zeros(data.x.shape[0], dtype=torch.int64))}
# ort_outs = torch.tensor(ort_session.run(None, ort_inputs))
# import torch.nn.functional as F 
# probs = F.sigmoid(ort_outs)
# preds = (probs > 0.5).int()
# print(preds)
