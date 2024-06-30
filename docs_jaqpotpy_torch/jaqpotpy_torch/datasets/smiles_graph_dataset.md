Module jaqpotpy_torch.datasets.smiles_graph_dataset
===================================================
Author: Ioannis Pitoskas (jpitoskas@gmail.com)

Classes
-------

`SmilesGraphDataset(smiles, y=None, featurizer=None)`
:   A PyTorch Dataset class for handling SMILES strings as graphs.
    This class overrides `__getitem__` and `__len__` (check source code for methods' docstrings).
    
    Attributes:
        smiles (list): A list of SMILES strings.
        y (list, optional): A list of target values.
        featurizer (SmilesGraphFeaturizer): The object to transform SMILES strings into graph representations.
        precomputed_features (list, optional): A list of precomputed features. If precompute_featurization() is not called, this attribute remains None.
    
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

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic

    ### Descendants

    * jaqpotpy_torch.datasets.smiles_graph_dataset.SmilesGraphDatasetWithExternal

    ### Methods

    `get_atom_feature_labels(self)`
    :   Returns the atom feature labels.
        
        Returns:
            list: A list of atom feature labels.

    `get_bond_feature_labels(self)`
    :   Returns the bond feature labels.
        
        Returns:
            list: A list of bond feature labels.

    `get_num_edge_features(self)`
    :   Returns the number of edge features.
        
        Returns:
            int: Number of edge features.

    `get_num_node_features(self)`
    :   Returns the number of node features.
        
        Returns:
            int: Number of node features.

    `precompute_featurization(self)`
    :   Precomputes the featurization of the dataset.

`SmilesGraphDatasetWithExternal(smiles, external, y=None, featurizer=None)`
:   A PyTorch Dataset class for handling SMILES strings as graphs, along with additional external features.
    This class inherits from SmilesGraphDataset and overrides `__getitem__` and `__len__` (check source code for methods' docstrings).
    
    Attributes:
        smiles (list): A list of SMILES strings.
        y (list, optional): A list of target values.
        featurizer (SmilesGraphFeaturizer): The object to transform SMILES strings into graph representations.
        precomputed_features (list, optional): A list of precomputed features. If precompute_featurization() is not called, this attribute remains None.
        external (torch.Tensor): A 2D tensor containing the external features.
    
    The SmilesGraphDatasetWithExternal constructor.
    
    Args:
        smiles (list): A list of SMILES strings.
        external (numpy.ndarray or pandas.DataFrame): External feature data 2D matrix.
        y (list, optional): A list of target values. Default is None.
        featurizer (SmilesGraphFeaturizer, optional): A featurizer object for to create graph representations from SMILES strings. Default is None.

    ### Ancestors (in MRO)

    * jaqpotpy_torch.datasets.smiles_graph_dataset.SmilesGraphDataset
    * torch.utils.data.dataset.Dataset
    * typing.Generic

    ### Methods

    `get_num_external_features(self)`
    :   Returns the number of external features.
        
        Returns:
            int: Number of external features.