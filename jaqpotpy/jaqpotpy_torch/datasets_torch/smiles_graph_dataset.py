from torch.utils.data import Dataset
from ..featurizers_torch import SmilesGraphFeaturizer


class SmilesGraphDataset(Dataset):

    def __init__(self, smiles, y=None, featurizer=None):
        super(SmilesGraphDataset, self).__init__()
        
        self.smiles = smiles
        self.y = y
        
        if featurizer:
            self.featurizer = featurizer
        else:
            self.featurizer = SmilesGraphFeaturizer()
            self.featurizer.set_default_config()
        
        self.precomputed_features = None
    
    
    def config_from_other_dataset(self, dataset):
        self.featurizer = SmilesGraphFeaturizer().config_from_other_featurizer(dataset.featurizer)
        return self

    
    def get_atom_feature_labels(self):        
        return self.featurizer.get_atom_feature_labels()
    
    
    def get_bond_feature_labels(self):
        return self.featurizer.get_bond_feature_labels()
        
        
    def precompute_featurization(self):
        if self.y:
            self.precomputed_features = [self.featurizer(sm, y) for sm, y, in zip(self.smiles, self.y)]
        else:
            self.precomputed_features = [self.featurizer(sm) for sm in self.smiles]

        
    def get_num_node_features(self):
        return len(self.get_atom_feature_labels())
    
    
    def get_num_edge_features(self):
        return len(self.get_bond_feature_labels())

    
    def __getitem__(self, idx):
        
        if self.precomputed_features:
            return self.precomputed_features[idx]
        
        sm = self.smiles[idx]
        y = self.y[idx] if self.y else None

        
        return self.featurizer(sm, y)

    
    def __len__(self):
        return len(self.smiles)
