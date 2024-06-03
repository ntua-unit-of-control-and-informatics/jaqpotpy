from torch.utils.data import Dataset
from ..featurizers import SmilesGraphFeaturizer
import torch

class SmilesGraphDataset(Dataset):

    def __init__(self, smiles, y=None, featurizer=None):
        super().__init__()
        
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




class SmilesGraphDatasetWithExternal(SmilesGraphDataset):
    def __init__(self, smiles, external, y=None, featurizer=None, external_normalization_mean=None, external_normalization_std=None):
        super().__init__(smiles, y=y, featurizer=featurizer)


        self.external = torch.tensor(external)
        if self.external.ndim != 2:
            raise ValueError("External data must be a 2D array.")

        if self.external.shape[0] != len(smiles):
            raise ValueError("External data must have the same number of rows as smiles strings.")


        self.external_feature_labels = None


        if external_normalization_mean is None:
            self.external_normalization_mean = torch.zeros(self.external.shape[1])
        else:
            self.external_normalization_mean = torch.tensor(external_normalization_mean)
            if self.external.ndim != 1:
                raise ValueError("external_normalization_mean must be a 1D array.")
            if self.external_normalization_mean.shape[0] != external.shape[1]:
                raise ValueError("external_normalization_mean must have the same number of elements as external features")


        if external_normalization_std is None:
            self.external_normalization_std = torch.ones(self.external.shape[1])
        else:
            self.external_normalization_std = torch.tensor(external_normalization_std)
            if self.external.ndim != 1:
                raise ValueError("external_normalization_std data must be a 1D array.")
            if self.external_normalization_std.shape[0] != external.shape[1]:
                raise ValueError("external_normalization_std must have the same number of elements as external features")


    def set_external_feature_labels(self, external_feature_labels):
        if len(external_feature_labels) != self.external.size(1):
            raise ValueError("Number of external feature labels must match the number of columns in the external data.")
        self.external_feature_labels = list(external_feature_labels)

    def get_external_feature_labels(self):
        return self.external_feature_labels

    def _normalize_external(self, external_row):
        normalized_external_row = (external_row - self.external_normalization_mean) / self.external_normalization_std
        return normalized_external_row

    def __getitem__(self, idx):

        if self.precomputed_features:
            return self.precomputed_features[idx]

        sm = self.smiles[idx]
        y = self.y[idx] if self.y else None
        external = self.external[idx]

        data = self.featurizer(sm, y)
        data.external = self._normalize_external(external_row=external)
        return data
