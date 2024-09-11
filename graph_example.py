smiles_train = [
    "COc1cccc(S(=O)(=O)Cc2cc(C(=O)NO)no2)c1",
    "O=C(N[C@H](CO)[C@H](O)c1ccc([N+](=O)[O-])cc1)C(Cl)Cl",
    "CC(=O)NC[C@H]1CN(c2ccc3c(c2F)CCCCC3=O)C(=O)O1",
    "CC(C)(C)OC(=O)C=C",
    "CC1(CC(CC(C1)(C)CN=C=O)N=C=O)C",
    "C1=CN=CN1",
]

smiles_val = [
    "CC(=O)NC[C@H]1CN(c2ccc(C(C)=O)cc2)C(=O)O1",
    "NCCNC(=O)COc1c2cccc1Cc1cccc(c1O)Cc1cccc(c1OCC(=O)NCCN)Cc1cccc(c1O)C2",
    "C1=CC=CC=C1",
    "CCCCCCC",
    "OP(=O)(O)[O-].[Na+]",
]
import warnings

warnings.simplefilter(action="ignore")
# Binary labels
y_train = [0, 1, 1, 0, 1, 1]
y_val = [1, 0, 0, 0, 1]

from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer
from rdkit import Chem

featurizer = SmilesGraphFeaturizer()

featurizer.add_atom_feature(
    "symbol", ["C", "O", "N", "Cl", "S", "F", "Na", "P", "Br", "Si", "K", "Sn", "UNK"]
)
featurizer.add_atom_feature("degree", [0, 1, 2, 3, 4])
featurizer.add_atom_feature(
    "hybridization",
    [
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
    ],
)
featurizer.add_atom_feature("implicit_valence", [0, 1, 2, 3, 4])
featurizer.add_atom_feature("is_aromatic")

first = featurizer.featurize(smiles_train[0])
print(first)

# from jaqpotpy.datasets.graph_pyg_dataset import SmilesGraphDataset

# train_dataset = SmilesGraphDataset(smiles_train, y_train, featurizer=featurizer)
# val_dataset = SmilesGraphDataset(smiles_val, y_val, featurizer=featurizer)

# print(f"Train dataset consists of {len(train_dataset)} samples.")
# print(f"Val dataset consists of {len(val_dataset)} samples.")

# train_dataset.precompute_featurization()
# val_dataset.precompute_featurization()

# from jaqpotpy.models.torch_geometric_models.graph_convolutional_network import (
#     GraphConvolutionNetwork,
# )
# from jaqpotpy.models.torch_geometric_models.graph_sage_network import GraphSageNetwork

# # from jaqpotpy.models.torch_geometric_models.graph_attention_network import  GraphAttentionNetwork

# input_dim = featurizer.get_num_node_features()
# print(input_dim)
# model = GraphConvolutionNetwork(
#     input_dim, hidden_layers=2, hidden_dim=16, output_dim=1, graph_norm=False
# )
# import torch
# import torch.nn as nn

# n_epochs = 3
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
# loss_fn = nn.BCEWithLogitsLoss()

# # DEVICE
# seed = 42
# use_gpu = True  # Change this for use of CPU or GPU (if available)
# torch.manual_seed(seed)
# if torch.cuda.is_available() and use_gpu:
#     device = torch.device("cuda:0")
#     torch.cuda.manual_seed(seed)
#     print(f"\nDevice: \n- {torch.cuda.get_device_name(device=device)}")
# else:
#     device = torch.device("cpu")
#     print("\nDevice: 'CPU'")

# from jaqpotpy.models.trainers.binary_trainers import BinaryGraphModelTrainer

# from torch_geometric.loader import DataLoader

# train_loader = DataLoader(
#     train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True
# )
# val_loader = DataLoader(
#     val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True
# )

# trainer = BinaryGraphModelTrainer(
#     model,
#     n_epochs=n_epochs,
#     optimizer=optimizer,
#     loss_fn=loss_fn,
#     device=device,
#     log_enabled=True,
# )

# trainer.train(train_loader, val_loader=val_loader)

# # depl_json = trainer.prepare_for_deployment(featurizer=featurizer,
# #                                 endpoint_name='Y',
# #                                 name='SMILES-Binary-Classifier')

# smiles = "CC(=O)OCCC(/C)=C\C[C@H](C(C)=C)CCC=C"
# data = featurizer(smiles)
# model.eval()
# torch_preds = []
# with torch.no_grad():
#     outputs = model(x=data.x, edge_index=data.edge_index, batch=data.batch)
#     outputs = outputs.squeeze(-1)
#     import torch.nn.functional as F

#     probs = F.sigmoid(outputs)
#     preds = (probs > 0.5).int()
# print("Pytorch geometric Probability", probs)
# print("Pytorch geometric Predictions", preds)
# # print(type(depl_json['additional_model_params']))
# # atom_labels = featurizer.get_atom_feature_labels()
# # print(atom_labels)

# # model = depl_json['actualModel']
# # # print(model)
# # import base64
# # onnx_model = base64.b64decode(model)
# # import onnxruntime
# # ort_session = onnxruntime.InferenceSession(onnx_model)
# # feat_config = depl_json['additional_model_params']['featurizer']
# # featurizer = SmilesGraphFeaturizer()
# # featurizer.load_json_rep(feat_config)

# # def to_numpy(tensor):
# #         return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# # data = featurizer(smiles)
# # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data.x),
# #             ort_session.get_inputs()[1].name: to_numpy(data.edge_index),
# #             ort_session.get_inputs()[2].name: to_numpy(torch.zeros(data.x.shape[0], dtype=torch.int64))}
# # import numpy as np
# # ort_outs = torch.tensor(np.array(ort_session.run(None, ort_inputs)))
# # import torch.nn.functional as F
# # probs = F.sigmoid(ort_outs)
# # preds = (probs > 0.5).int()
# # print(probs)

# from jaqpotpy.jaqpot import Jaqpot

# onnx_model = trainer.pyg_to_onnx(featurizer)
# jaqpot = Jaqpot(base_url="http://localhost.jaqpot.org:8080/")
# jaqpot.set_api_key(
#     "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI5LVBxT3loU19xTmYtaDFzNmQ5SXZzN29PMDM5dmEzeElKeURsOTZpd2ZRIn0.eyJleHAiOjE3MjU1NzM2ODgsImlhdCI6MTcyNTUzNzY4OCwianRpIjoiZDNlMWJiYjYtYjI5Yi00ZGM5LTgwODctODgxOGJhOThjNDFjIiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdC5qYXFwb3Qub3JnOjgwNzAvcmVhbG1zL2phcXBvdC1sb2NhbCIsImF1ZCI6ImFjY291bnQiLCJzdWIiOiJiZWM0NDkzNS04ZjU2LTRiOTgtYThiYi0wZTdlNWUwYjA5N2EiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJqYXFwb3QtbG9jYWwtdGVzdCIsInNlc3Npb25fc3RhdGUiOiJmYzA0OTI0OS02NDQ3LTRlMGItYWNhMC04ZWViMWI0YjFhNzkiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIiIsIioiXSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoiZW1haWwgcHJvZmlsZSIsInNpZCI6ImZjMDQ5MjQ5LTY0NDctNGUwYi1hY2EwLThlZWIxYjRiMWE3OSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYW1lIjoiamFxIHBvdCIsInByZWZlcnJlZF91c2VybmFtZSI6ImphcXBvdCIsImdpdmVuX25hbWUiOiJqYXEiLCJmYW1pbHlfbmFtZSI6InBvdCIsImVtYWlsIjoiamFxcG90QGphcXBvdC5vcmcifQ.W-4e9ut3B9FKNs3Hubc6-w_xdVp4vkDR0Y3KMa-y29Z6kz9DWzyKBjLEWjYOAiHH-IeTOUs2pqp6RkkmXb1K0vLnv4GDM-1KjRMvxdcztt5q3ek0WDnKJb7R3vTulYeRZ2wwIQLoD6CP2lcG6jb7oNacCuUAVg3UimayR3HcRZCiu-gkVSAWwLsAZsFUS7nC7tErqhLoIDTLPN3C12-3FFBZrumXt-IQxBrfTUnJUvAphDBB0OOPSuSrEvgfA5kgEs2rFoJEUiyHLZMr4faPT1YqBQ4t9yYPDC1QyFxb8xZQVRvvIb6JutPEEtGeETyvhCRVZCegP1SOOZndoLPatg"
# )
# jaqpot.deploy_Torch_Graph_model(
#     onnx_model=onnx_model,
#     featurizer=featurizer,
#     name="Remove Tests",
#     description="Binary GCN classification with node features",
#     target_name="Random",
#     visibility="PRIVATE",
# )
