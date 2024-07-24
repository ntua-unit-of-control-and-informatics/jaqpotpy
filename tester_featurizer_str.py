from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer
from jaqpotpy.datasets.graph_pyg_dataset import SmilesGraphDataset
from rdkit import Chem

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
y_train = [0, 1, 0, 1, 1, 0]
y_val = [1, 0, 1, 0, 1]

featurizer = SmilesGraphFeaturizer()
featurizer.add_atom_feature('symbol', ['C', 'O', 'N', 'F', 'Cl', 'Br', 'I', 'UNK'])
featurizer.add_atom_feature('degree', [0, 1, 2, 3, 4])
featurizer.add_atom_feature('hybridization', [Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3,
                        Chem.rdchem.HybridizationType.S,
                        Chem.rdchem.HybridizationType.SP])
featurizer.add_atom_feature('implicit_valence', [0, 1, 2, 3, 4])
# Vectors that are not one hot-encoded do not need the possible values
featurizer.add_atom_feature('formal_charge')
featurizer.add_atom_feature('is_aromatic')
featurizer.add_atom_feature('num_radical_electrons')
featurizer.add_atom_feature('_ChiralityPossible')
featurizer.add_atom_feature('mass')
featurizer.add_bond_feature('bond_type',[Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC])
featurizer.add_bond_feature('is_conjugated')

feat_json = featurizer.get_json_rep()

from jaqpotpy.datasets.graph_pyg_dataset import SmilesGraphDataset

train_dataset = SmilesGraphDataset(smiles_train, y_train, featurizer=featurizer)
val_dataset = SmilesGraphDataset(smiles_val, y_val, featurizer=featurizer)

train_dataset.precompute_featurization()
val_dataset.precompute_featurization()

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


from jaqpotpy.models.torch_geometric_models.graph_convolutional_network import GraphConvolutionalNetwork
from jaqpotpy.models.torch_geometric_models.graph_transformer_network import GraphTransformerNetwork
from jaqpotpy.models.torch_geometric_models.graph_sage_network import GraphSAGENetwork
from jaqpotpy.models.torch_geometric_models.graph_attention_network import GraphAttentionNetwork
input_dim = featurizer.get_num_node_features()
edge_dim = featurizer.get_num_edge_features()
# edge_dim = None
model = GraphConvolutionalNetwork(input_dim=input_dim,
                                hidden_dims=[2, 2],
                                output_dim=1,
                                dropout=0.5,
                                graph_norm=False)

import torch
import torch.nn as nn

n_epochs = 1
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()

from jaqpotpy.models.trainers.regression_trainers.regression_graph_model_trainer import RegressionGraphModelTrainer
from jaqpotpy.models.trainers.binary_trainers.binary_graph_model_trainer import BinaryGraphModelTrainer

trainer = BinaryGraphModelTrainer(model,
                                  n_epochs=n_epochs,
                                  optimizer=optimizer,
                                  loss_fn=loss_fn,
                                  log_enabled=True)


trainer.train(train_loader, val_loader=val_loader)

model_data = trainer.prepare_for_deployment(featurizer=featurizer,
                               endpoint_name='demo_binary_var',
                               name='SMILES-Binary-Classifier')


import base64
import io
import pickle
import torch.nn.functional as F

print(feat_json)
new_featurizer = SmilesGraphFeaturizer()
new_featurizer.load_json_rep(feat_json)
train_dataset = SmilesGraphDataset(smiles_train, y_train, featurizer= new_featurizer)
val_dataset = SmilesGraphDataset(smiles_val, y_val, featurizer= new_featurizer)

train_dataset.precompute_featurization()
val_dataset.precompute_featurization()

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

model.eval()
torch_preds = []
for data in val_loader:
    with torch.no_grad():
        outputs = model(x=data.x, edge_index=data.edge_index, batch=data.batch)
        outputs = outputs.squeeze(-1)
        probs = F.sigmoid(outputs)
        preds = (probs > 0.5).int()
    torch_preds.append(preds)
    #torch_preds.append(outputs)



import onnxruntime
ort_session = onnxruntime.InferenceSession("model.onnx")

def to_numpy(tensor):
     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
onnx_preds = []
for data in val_loader:
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data.x),
                  ort_session.get_inputs()[1].name: to_numpy(data.edge_index),
                  ort_session.get_inputs()[2].name: to_numpy(torch.zeros(data.x.shape))}
    ort_outs = torch.tensor(ort_session.run(None, ort_inputs))
    probs = F.sigmoid(ort_outs)
    preds = (probs > 0.5).int()
    onnx_preds.append(preds.squeeze(-1).squeeze(-1))
print(torch_preds)
print(onnx_preds)
#featurizer_buffer = io.BytesIO()
#pickle.dump(featurizer, featurizer_buffer)
#featurizer_buffer.seek(0)
#featurizer_pickle_base64 = base64.b64encode(featurizer_buffer.getvalue()).decode('utf-8')

