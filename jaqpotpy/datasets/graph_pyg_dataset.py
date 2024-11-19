from torch.utils.data import Dataset
from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer
from typing import Any, Optional


class SmilesGraphDataset(Dataset):
    """
    A PyTorch Dataset class for handling SMILES strings as graph data suitable for training
    graph neural networks. The class transforms SMILES strings into graph representations using
    a specified featurizer and optionally supports target values for supervised learning tasks.

    Attributes:
        smiles (list): A list of SMILES strings to be converted into graph data.
        y (list, optional): A list of target values associated with each SMILES string.
        featurizer (SmilesGraphFeaturizer): A featurizer to transform SMILES into graph representations.
        precomputed_features (list, optional): Precomputed graph features; remains None until `precompute_featurization` is called.
    """

    def __init__(
        self,
        smiles: list = None,
        y: Optional[list] = None,
        featurizer: Optional[SmilesGraphFeaturizer] = None,
    ):
        """
        Initializes the SmilesGraphDataset with SMILES strings, target values, and an optional featurizer.

        Args:
            smiles (list): List of SMILES strings to be transformed into graphs.
            y (list, optional): List of target values for supervised learning tasks.
            featurizer (SmilesGraphFeaturizer, optional): A featurizer for converting SMILES into graphs;
                                                         if not provided, a default featurizer is used.
        """
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
        """
        Precomputes the featurized graph representations of the SMILES strings in the dataset.

        This method prepares the graph data in advance, which can improve efficiency when
        accessing individual data samples. Each SMILES string is transformed into a graph
        representation (and paired with its target value if available).

        Sets:
            self.precomputed_features (list): A list of graph features precomputed for each SMILES.
        """
        if self.y:
            self.precomputed_features = [
                self.featurizer(sm, y) for sm, y in zip(self.smiles, self.y)
            ]
        else:
            self.precomputed_features = [self.featurizer(sm) for sm in self.smiles]

    def get_num_node_features(self):
        """
        Returns the number of node features (atom-level features) in each graph representation.

        Returns:
            int: The number of features associated with each node (atom).
        """
        return len(self.get_atom_feature_labels())

    def get_num_edge_features(self):
        """
        Returns the number of edge features (bond-level features) in each graph representation.

        Returns:
            int: The number of features associated with each edge (bond).
        """
        return len(self.get_bond_feature_labels())

    def __getitem__(self, idx):
        """
        Retrieves a featurized graph representation and target value (if available) for a specific index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch_geometric.data.Data: A graph data object containing:
                - x (torch.Tensor): Node feature matrix with shape [num_nodes, num_node_features].
                - edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].
                - edge_attr (torch.Tensor, optional): Edge feature matrix with shape [num_edges, num_edge_features].
                - y (float, optional): Target value associated with the graph, if provided.
                - smiles (str): The SMILES string for the specific sample.
        """
        if self.precomputed_features:
            return self.precomputed_features[idx]
        sm = self.smiles[idx]
        y = self.y[idx] if self.y else None
        return self.featurizer(sm, y)

    def __len__(self):
        """
        Returns the total number of SMILES strings in the dataset, necessary for data loading operations.

        Returns:
            int: The length of the dataset (number of SMILES strings).
        """
        return len(self.smiles)

    def __repr__(self) -> str:
        """
        Returns a formal string representation of the SmilesGraphDataset object, indicating its class name.

        Returns:
            str: Class name as the representation of the dataset object.
        """
        return self.__class__.__name__
