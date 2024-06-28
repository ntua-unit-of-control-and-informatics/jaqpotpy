"""
Author: Ioannis Pitoskas (jpitoskas@gmail.com)
"""

from torch.utils.data import Dataset
from ..featurizers.smiles_graph_featurizer import SmilesGraphFeaturizer
import torch
import pandas as pd


class SmilesGraphDataset(Dataset):
    """
    A PyTorch Dataset class for handling SMILES strings as graphs.
    This class overrides `__getitem__` and `__len__` (check source code for methods' docstrings).

    Attributes:
        smiles (list): A list of SMILES strings.
        y (list, optional): A list of target values.
        featurizer (SmilesGraphFeaturizer): The object to transform SMILES strings into graph representations.
        precomputed_features (list, optional): A list of precomputed features. If precompute_featurization() is not called, this attribute remains None.
    """

    def __init__(self, smiles, y=None, featurizer=None):
        """
        The SmilesGraphDataset constructor.

        Args:
            smiles (list): A list of SMILES strings. 
            y (list, optional): A list of target values. Default is None.
            featurizer (SmilesGraphFeaturizer, optional): A featurizer object for to create graph representations from SMILES strings.
        
        Example:
        ```
        >>> from jaqpotpy.jaqpotpy_torch.featurizers import SmilesGraphFeaturizer
        >>> from rdkit import Chem
        >>>
        >>> smiles = ['C1=CN=CN1', 'CCCCCCC']
        >>> y = [0, 1]
        >>> featurizer = SmilesGraphFeaturizer()
        >>> featurizer.add_atom_characteristic('symbol', ['C', 'O', 'Na', 'Cl', 'UNK'])
        >>> featurizer.add_atom_characteristic('is_in_ring')
        >>> featurizer.add_bond_characteristic('bond_type', [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE])
        >>>
        >>> dataset = SmilesGraphDataset(smiles, y=y, featurizer=featurizer)
        >>> dataset[0]
        Data(x=[5, 6], edge_index=[2, 10], edge_attr=[10, 3], y=0, smiles='C1=CN=CN1')
        ```
        """


        super().__init__()
        
        self.smiles = smiles
        self.y = y
        
        if featurizer:
            self.featurizer = featurizer
        else:
            self.featurizer = SmilesGraphFeaturizer()
            self.featurizer.set_default_config()
        
        self.precomputed_features = None


    # def config_from_other_dataset(self, dataset):
    #     """
    #     Configures the dataset instance from another dataset.

    #     Args:
    #         dataset (SmilesGraphDataset): Another dataset to copy the configuration.

    #     Returns:
    #         SmilesGraphDataset: The configured dataset.
    #     """
    #     self.featurizer = SmilesGraphFeaturizer().config_from_other_featurizer(dataset.featurizer)
    #     return self


    def get_atom_feature_labels(self):
        """
        Returns the atom feature labels.

        Returns:
            list: A list of atom feature labels.
        """
        return self.featurizer.get_atom_feature_labels()


    def get_bond_feature_labels(self):
        """
        Returns the bond feature labels.

        Returns:
            list: A list of bond feature labels.
        """
        return self.featurizer.get_bond_feature_labels()


    def precompute_featurization(self):
        """
        Precomputes the featurization of the dataset.
        """
        if self.y:
            self.precomputed_features = [self.featurizer(sm, y) for sm, y, in zip(self.smiles, self.y)]
        else:
            self.precomputed_features = [self.featurizer(sm) for sm in self.smiles]


    def get_num_node_features(self):
        """
        Returns the number of node features.

        Returns:
            int: Number of node features.
        """
        return len(self.get_atom_feature_labels())


    def get_num_edge_features(self):
        """
        Returns the number of edge features.

        Returns:
            int: Number of edge features.
        """
        return len(self.get_bond_feature_labels())


    def __getitem__(self, idx):
        """
        Retrieves the featurized graph Data object (and target value) for a given index.

        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            torch_geometric.data.Data: A torch_geometric.data.Data object for this single sample containing:
                - x (torch.Tensor): Node feature matrix with shape [num_nodes, num_node_features].
                - edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].
                - edge_attr (torch.Tensor, optional): Edge feature matrix with shape [num_edges, num_edge_features].
                - y (float): Graph-level ground-truth label.
                - smiles (str): The SMILES string corresponding to the particular sample.
        """
        if self.precomputed_features:
            return self.precomputed_features[idx]
        
        sm = self.smiles[idx]
        y = self.y[idx] if self.y else None

        
        return self.featurizer(sm, y)


    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.smiles)




class SmilesGraphDatasetWithExternal(SmilesGraphDataset):
    """
    A PyTorch Dataset class for handling SMILES strings as graphs, along with additional external features.
    This class inherits from SmilesGraphDataset and overrides `__getitem__` and `__len__` (check source code for methods' docstrings).

    Attributes:
        smiles (list): A list of SMILES strings.
        y (list, optional): A list of target values.
        featurizer (SmilesGraphFeaturizer): The object to transform SMILES strings into graph representations.
        precomputed_features (list, optional): A list of precomputed features. If precompute_featurization() is not called, this attribute remains None.
        external (torch.Tensor): A 2D tensor containing the external features.
    """

    def __init__(self, smiles, external, y=None, featurizer=None):
        """
        The SmilesGraphDatasetWithExternal constructor.

        Args:
            smiles (list): A list of SMILES strings.
            external (numpy.ndarray or pandas.DataFrame): External feature data 2D matrix.
            y (list, optional): A list of target values. Default is None.
            featurizer (SmilesGraphFeaturizer, optional): A featurizer object for to create graph representations from SMILES strings. Default is None.
        """
        super().__init__(smiles, y=y, featurizer=featurizer)

        if isinstance(external, pd.DataFrame):
            external = external.to_numpy()

        self.external = torch.tensor(external).float()

        if self.external.ndim != 2:
            raise ValueError("External data must be a 2D array.")

        if self.external.shape[0] != len(smiles):
            raise ValueError("External data must have the same number of rows as smiles strings.")


    def get_num_external_features(self):
        """
        Returns the number of external features.

        Returns:
            int: Number of external features.
        """
        return self.external.size(1)
    

    def __getitem__(self, idx):
        """
        Retrieves the features and target value for a given index.

        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            torch_geometric.data.Data: A torch_geometric.data.Data object for this single sample containing:
                - x (torch.Tensor): Node feature matrix with shape [num_nodes, num_node_features].
                - edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].
                - edge_attr (torch.Tensor, optional): Edge feature matrix with shape [num_edges, num_edge_features].
                - y (float): Graph-level ground-truth label.
                - smiles (str): The SMILES string corresponding to the particular sample.
                - external (torch.Tensor): The external feature vector of the particular sample with shape (1, num_external_features)
        """

        # data = self.__getitem__(idx)
        if self.precomputed_features:
            data = self.precomputed_features[idx]
        else:
            sm = self.smiles[idx]
            y = self.y[idx] if self.y else None
            data = self.featurizer(sm, y)

        data.external = self.external[idx].unsqueeze(0)
        # data.external = self._normalize_external(external_row=external).unsqueeze(0)

        return data
    
    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return super().__len__()
