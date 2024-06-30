Module jaqpotpy_torch.featurizers.smiles_graph_featurizer
=========================================================
Author: Ioannis Pitoskas (jpitoskas@gmail.com)

Classes
-------

`SmilesGraphFeaturizer(include_edge_features=True, warnings_enabled=True)`
:   Featurizes SMILES strings into graph data suitable for graph neural networks.
    
    Attributes:
        include_edge_features (bool): Whether to include edge features in the featurization.
        warnings_enabled (bool): Whether to enable warnings.
        atom_allowable_sets (dict): Allowable sets for atom features.
        bond_allowable_sets (dict): Allowable sets for bond features.
    
    The SmilesGraphFeaturizer constructor.
    
    Args:
        include_edge_features (bool): Whether to include edge features in the featurization.
        warnings_enabled (bool): Whether to enable warnings.
    ```

    ### Ancestors (in MRO)

    * jaqpotpy_torch.featurizers.featurizer.Featurizer
    * abc.ABC

    ### Class variables

    `NON_ONE_HOT_ENCODED`
    :   list: Characteristics that are not categorical, and cannot be one-hot encoded

    `SUPPORTED_ATOM_CHARACTERISTICS`
    :   dict: Dictionary mapping atom characteristics to their respective featurization functions.

    `SUPPORTED_BOND_CHARACTERISTICS`
    :   dict: Dictionary mapping bond characteristics to their respective featurization functions.

    ### Methods

    `add_atom_characteristic(self, atom_characteristic, allowable_set=None)`
    :   Adds an atom characteristic to the featurizer.
        
        Args:
            atom_characteristic (str): The atom characteristic name to be added.
            allowable_set (list or None): The allowable set for the atom characteristic.
        
        Example:
        ```
        >>> featurizer = SmilesGraphFeaturizer()
        >>> featurizer.add_atom_characteristic('symbol', ['C', 'O', 'N', 'Cl']) # categorical characteristics need allowable set of categories
        >>> featurizer.add_atom_characteristic('is_in_ring') # allowable set is set to Nonefor non-categorical
        >>> featurizer.atom_allowable_sets
        {'symbol': ['C', 'O', 'N', 'Cl'], 'is_in_ring': None}
        ```
        
        Notes:
            'UNK' is used as a special keyword. When an atom characteristic is encoded and an unknown 
            value (not in the allowable set) is encountered during featurization, it will be encoded as 'UNK'.
            This allows handling of unexpected values by mapping them to a default or placeholder category.

    `add_bond_characteristic(self, bond_characteristic, allowable_set=None)`
    :   Adds a bond characteristic to the featurizer.
        
        Args:
            bond_characteristic (str): The bond characteristic to add.
            allowable_set (list or None): The allowable set for the bond characteristic.
        
        Example:
        ```
        >>> from rdkit import Chem
        >>> featurizer = SmilesGraphFeaturizer()
        >>> featurizer.add_bond_characteristic('bond_type', [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE])
        >>> featurizer.add_bond_characteristic('is_conjugated')
        >>> featurizer.bond_allowable_sets
        {'bond_type': [rdkit.Chem.rdchem.BondType.SINGLE,
        rdkit.Chem.rdchem.BondType.DOUBLE,
        rdkit.Chem.rdchem.BondType.TRIPLE],
        'is_conjugated': None}
        ```
        
        Notes:
            'UNK' is used as a special keyword. When a bond characteristic is encoded and an unknown 
            value (not in the allowable set) is encountered during featurization, it will be encoded as 'UNK'.
            This allows handling of unexpected values by mapping them to a default or placeholder category.

    `adjacency_matrix(self, mol)`
    :   Computes the adjacency matrix for a molecule.
        
        Args:
            mol (rdkit.Chem.Mol): The molecule to compute the adjacency matrix for.
        
        Returns:
            torch.Tensor: The adjacency matrix.

    `atom_features(self, atom)`
    :   Extracts features for an atom.
        
        Args:
            atom (rdkit.Chem.Atom): The atom to extract features from.
        
        Returns:
            list: The features for the atom.

    `bond_features(self, bond)`
    :   Extracts features for a bond.
        
        Args:
            bond (rdkit.Chem.Bond): The bond to extract features from.
        
        Returns:
            list: The features for the bond.

    `enable_warnings(self, enable=True)`
    :   Enables or disables warnings.
        
        Args:
            enable (bool): Whether to enable warnings.

    `extract_molecular_features(self, mol)`
    :   Extracts molecular features from a molecule.
        
        Args:
            mol (rdkit.Chem.Mol): The molecule to extract features from.
        
        Returns:
            tuple: The atom features and bond features as tensors.

    `featurize(self, sm, y=None)`
    :   Featurizes a SMILES string into graph data.
        
        Args:
            sm (str): The SMILES string.
            y: The target value (optional).
        
        Returns:
            torch_geometric.data.Data: The featurized data.

    `get_atom_feature_labels(self)`
    :   Returns the labels for the atom features.
        
        Returns:
            list: The labels for the atom features.

    `get_bond_feature_labels(self)`
    :   Returns the labels for the bond features.
        
        Returns:
            list: The labels for the bond features.

    `get_default_atom_allowable_set(self, atom_characteristic)`
    :   Returns the default allowable set for an atom characteristic.
        
        Args:
            atom_characteristic (str): The atom characteristic.
        
        Returns:
            list: The default allowable set for the atom characteristic.

    `get_default_bond_allowable_set(self, bond_characteristic)`
    :   Returns the default allowable set for a bond characteristic.
        
        Args:
            bond_characteristic (str): The bond characteristic.
        
        Returns:
            list: The default allowable set for the bond characteristic.

    `get_num_edge_features(self)`
    :   Returns the number of edge features.
        
        Returns:
            int: The number of edge features.

    `get_num_node_features(self)`
    :   Returns the number of node features.
        
        Returns:
            int: The number of node features.

    `get_supported_atom_characteristics(self)`
    :   Returns the names of the supported atom characteristics.

    `get_supported_bond_characteristics(self)`
    :   Returns the names of the supported bond characteristics.

    `load_config(self, config_file)`
    :   Loads the featurizer configuration from a file.
        
        Args:
            config_file (str): The file to load the configuration from.
        
        Returns:
            SmilesGraphFeaturizer: The configured featurizer.

    `one_of_k_encoding(self, x, allowable_set, characteristic_name=None)`
    :   One-hot encodes a value based on an allowable set.
        
        Args:
            x: The value to encode.
            allowable_set (list): The allowable set for the value.
            characteristic_name (str): The name of the characteristic (optional).
        
        Returns:
            list: The one-hot encoded vector.

    `one_of_k_encoding_unk(self, x, allowable_set)`
    :   One-hot encodes a value based on an allowable set, mapping unknown values to 'UNK'.
        
        Args:
            x: The value to encode.
            allowable_set (list): The allowable set for the value.
        
        Returns:
            list: The one-hot encoded value.

    `save(self, filepath='featurizer.pkl')`
    :   Saves the featurizer to a file.
        
        Args:
            filepath (str): The file to save the featurizer to.

    `save_config(self, config_file='featurizer_config.pkl')`
    :   Saves the featurizer configuration to a file.
        
        Args:
            config_file (str): The file to save the configuration to.

    `set_atom_allowable_sets(self, atom_allowable_sets_dict)`
    :   Sets the allowable sets for atom characteristics.
        
        Args:
            atom_allowable_sets_dict (dict): Dictionary of allowable sets per atom characteristic.

    `set_bond_allowable_sets(self, bond_allowable_sets_dict)`
    :   Sets the allowable sets for bond characteristics.
        
        Args:
            bond_allowable_sets_dict (dict): Dictionary of allowable sets per bond characteristic.

    `set_default_config(self)`
    :   Sets the default configuration for the featurizer.

    `warning(self, message)`
    :   Issues a warning if warnings are enabled.
        
        Args:
            message (str): The warning message.