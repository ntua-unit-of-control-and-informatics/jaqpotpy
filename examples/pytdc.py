from tdc.single_pred import HTS

from jaqpotpy.datasets import TorchGraphDataset
from jaqpotpy.descriptors.molecular import MolGraphConvFeaturizer
from jaqpotpy.models import Evaluator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
from jaqpotpy.models import GCN_V1, MolecularTorchGeometric


data = HTS(name='SARSCoV2_3CLPro_Diamond')
split = data.get_split()

train_mols = split['train']['Drug']
train_y = split['train']['Y']
test_mols = split['test']['Drug']
test_y = split['test']['Y']


featurizer = MolGraphConvFeaturizer(use_partial_charge=True, use_chirality=True)
train_dataset = TorchGraphDataset(smiles=train_mols, y=train_y, task='classification'
                            , featurizer=featurizer)

test_dataset = TorchGraphDataset(smiles=test_mols, y=test_y, task='classification'
                            , featurizer=featurizer)

train_dataset.create()
test_dataset.create()

val = Evaluator()
val.dataset = test_dataset
val.register_scoring_function('Accuracy', accuracy_score)
val.register_scoring_function('F1', f1_score)
val.register_scoring_function('Roc Auc', roc_auc_score)
model = GCN_V1(33, 3, 40, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
m = MolecularTorchGeometric(dataset=train_dataset
                            , model_nn=model, eval=val
                            , train_batch=180, test_batch=80
                            , epochs=340, optimizer=optimizer, criterion=criterion).fit()

m.eval()
# model = m.create_molecular_model()
