from torch.utils.data import Dataset
from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer
from typing import Any, Optional


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

    def __init__(
        self,
        smiles: list = None,
        y: Optional[list] = None,
        featurizer: Optional[SmilesGraphFeaturizer] = None,
    ):
        """The SmilesGraphDataset constructor."""
        super().__init__()
        self.smiles = smiles
        self.y = y
        # For the moment there is only one Featurizer for Graph.
        if featurizer:
            self.featurizer = featurizer
            self.featurizer.sort_allowable_sets()
        else:
            self.featurizer = SmilesGraphFeaturizer()
            # Default node, edge features in case of not specifying dataset
            self.featurizer.set_default_config()
            self.featurizer.sort_allowable_sets()
        self.precomputed_features = None

    def precompute_featurization(self):
        """Precomputes the featurization of the dataset before being accessed by __getitem__"""
        if self.y:
            self.precomputed_features = [
                self.featurizer(sm, y) for sm, y, in zip(self.smiles, self.y)
            ]
        else:
            self.precomputed_features = [self.featurizer(sm) for sm in self.smiles]

    def get_num_node_features(self):
        """Returns the number of node features."""
        return len(self.get_atom_feature_labels())

    def get_num_edge_features(self):
        """Returns the number of edge features."""
        return len(self.get_bond_feature_labels())

    def __getitem__(self, idx):
        """
        Retrieves the featurized graph Data object and target value for a given index.
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
        __len__ functionality is important for the DataLoader to determine batching,
        shuffling and iterating over the dataset.
        """
        return len(self.smiles)

    def __repr__(self) -> str:
        """Official string representation of the Dataset Object"""
        return self.__class__.__name__
