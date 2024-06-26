"""
Author: Ioannis Pitoskas
Contact: jpitoskas@gmail.com
"""

from rdkit import Chem
import torch
from torch_geometric.data import Data
import warnings
import pickle
import json

from .featurizer import Featurizer

class SmilesGraphFeaturizer(Featurizer):
    """
    Featurizes SMILES strings into graph data suitable for graph neural networks.

    Attributes:
        include_edge_features (bool): Whether to include edge features in the featurization.
        warnings_enabled (bool): Whether to enable warnings.
        atom_allowable_sets (dict): Allowable sets for atom features.
        bond_allowable_sets (dict): Allowable sets for bond features.
    """
    
    SUPPORTED_ATOM_CHARACTERISTICS = {"symbol": lambda atom: atom.GetSymbol(),
                                      "degree": lambda atom: atom.GetDegree(),
                                      "total_degree": lambda atom: atom.GetTotalDegree(),
                                      "formal_charge": lambda atom: atom.GetFormalCharge(),
                                      "num_radical_electrons": lambda atom: atom.GetNumRadicalElectrons(),
                                      "hybridization": lambda atom: atom.GetHybridization(),
                                      "is_aromatic": lambda atom: atom.GetIsAromatic(),
                                      "is_in_ring": lambda atom: atom.IsInRing(),
                                      "total_num_hs": lambda atom: atom.GetTotalNumHs(),
                                      "num_explicit_hs": lambda atom: atom.GetNumExplicitHs(),
                                      "num_implicit_hs": lambda atom: atom.GetNumImplicitHs(),
                                      "_ChiralityPossible": lambda atom: atom.HasProp('_ChiralityPossible'),
                                      "isotope": lambda atom: atom.GetIsotope(),
                                      "total_valence": lambda atom: atom.GetTotalValence(),
                                      "explicit_valence": lambda atom: atom.GetExplicitValence(),
                                      "implicit_valence": lambda atom: atom.GetImplicitValence(),
                                      "chiral_tag": lambda atom: atom.GetChiralTag(), 
                                      "mass": lambda atom: (atom.GetMass()-14.5275)/9.4154}

    SUPPORTED_BOND_CHARACTERISTICS = {"bond_type": lambda bond: bond.GetBondType(),
                                      "is_conjugated": lambda bond: bond.GetIsConjugated(),
                                      "is_in_ring": lambda bond: bond.IsInRing(),
                                      "stereo": lambda bond: bond.GetStereo()}
    
    NON_ONE_HOT_ENCODED = [
        # "formal_charge",
                           "num_radical_electrons",
                           "is_aromatic",
                           "_ChiralityPossible",
                           "is_conjugated",
                           "is_in_ring",
                           "mass"]

    def __init__(self, include_edge_features=True, warnings_enabled=True):
        """
        Initializes the SmilesGraphFeaturizer.

        Args:
            include_edge_features (bool): Whether to include edge features in the featurization.
            warnings_enabled (bool): Whether to enable warnings.
        """
        
        self.include_edge_features = include_edge_features
        
        self.warnings_enabled = warnings_enabled
        
        self.atom_allowable_sets = {}
        self.bond_allowable_sets = {}
        
    
    def get_supported_atom_characteristics(self):
        """Returns the names of the supported atom characteristics."""
        return self.SUPPORTED_ATOM_CHARACTERISTICS.keys()
    
    def get_supported_bond_characteristics(self):
        """Returns the names of the supported bond characteristics."""
        return self.SUPPORTED_BOND_CHARACTERISTICS.keys()
    
    def config_from_other_featurizer(self, featurizer):
        """
        Configures the featurizer from another SmilesGraphFeaturizer instance.

        Args:
            featurizer (Featurizer): Another SmilesGraphFeaturizer instance.

        Returns:
            SmilesGraphFeaturizer: The configured featurizer.
        """
        
        self.include_edge_features = featurizer.include_edge_features
        
        self.warnings_enabled = featurizer.warnings_enabled
        
        self.set_atom_allowable_sets(featurizer.atom_allowable_sets)

        self.set_bond_allowable_sets(featurizer.bond_allowable_sets)
        
        return self
        
    
    # def set_atom_characteristics(self, atom_characteristics):
    #     for key in atom_characteristics:
    #         if key not in self.SUPPORTED_ATOM_CHARACTERISTICS.keys(): 
    #             raise ValueError(f"Invalid atom characteristic '{key}'")
    #     self.atom_characteristics = list(atom_characteristics)
        
        
    # def set_bond_characteristics(self, bond_characteristics):
    #     for key in bond_characteristics:
    #         if key not in self.SUPPORTED_BOND_CHARACTERISTICS.keys(): 
    #             raise ValueError(f"Invalid bond characteristic '{key}'")
    #     self.bond_characteristics = list(bond_characteristics)
    
    
    def set_atom_allowable_sets(self, atom_allowable_sets_dict):
        """
        Sets the allowable sets for atom characteristics.

        Args:
            atom_allowable_sets_dict (dict): Dictionary of allowable sets per atom characteristic.
        """
        
        self.atom_allowable_sets = dict()
        for atom_characteristic, allowable_set in atom_allowable_sets_dict.items():
            self.add_atom_characteristic(atom_characteristic, allowable_set)
    
    def set_bond_allowable_sets(self, bond_allowable_sets_dict):
        """
        Sets the allowable sets for bond characteristics.

        Args:
            bond_allowable_sets_dict (dict): Dictionary of allowable sets per bond characteristic.
        """
        
        self.bond_allowable_sets = dict()
        for bond_characteristic, allowable_set in bond_allowable_sets_dict.items():
            self.add_bond_characteristic(bond_characteristic, allowable_set)

    def add_atom_characteristic(self, atom_characteristic, allowable_set=None):
        """
        Adds an atom characteristic to the featurizer.

        Args:
            atom_characteristic (str): The atom characteristic name to be added.
            allowable_set (list or None): The allowable set for the atom characteristic.
        """

        if atom_characteristic not in self.get_supported_atom_characteristics():
            raise ValueError(f"Unsupported atom characteristic '{atom_characteristic}'")
        
        if atom_characteristic in self.atom_allowable_sets.keys():
            self.warning(f"The atom allowable set for '{atom_characteristic}' will be overwritten.")

        if atom_characteristic in self.NON_ONE_HOT_ENCODED:
            if allowable_set is not None:
                self.warning(f"Atom allowable set given for '{atom_characteristic}' will be ignored (not one-hot encoded)")
            self.atom_allowable_sets[atom_characteristic] = None
        else:
            if allowable_set is None:
                self.warning(f"The atom allowable set for '{atom_characteristic}' is set to default.")
                allowable_set = self.get_default_atom_allowable_set(atom_characteristic)
            elif not isinstance(allowable_set, (list, tuple)):
                raise TypeError("Input dictionary must have values of type list, tuple or None.")
            self.atom_allowable_sets[atom_characteristic] = list(allowable_set)

    def add_bond_characteristic(self, bond_characteristic, allowable_set=None):
        """
        Adds a bond characteristic to the featurizer.

        Args:
            bond_characteristic (str): The bond characteristic to add.
            allowable_set (list or None): The allowable set for the bond characteristic.
        """

        if bond_characteristic not in self.get_supported_bond_characteristics():
            raise ValueError(f"Unsupported bond characteristic '{bond_characteristic}'")
        
        if bond_characteristic in self.bond_allowable_sets.keys():
            self.warning(f"The bond allowable set for '{bond_characteristic}' will be overwritten.")

        if bond_characteristic in self.NON_ONE_HOT_ENCODED:
            if allowable_set is not None:
                self.warning(f"Bond allowable set given for '{bond_characteristic}' will be ignored (not one-hot encoded)")
            self.bond_allowable_sets[bond_characteristic] = None
        else:
            if allowable_set is None:
                self.warning(f"The bond allowable set for '{bond_characteristic}' is set to default.")
                allowable_set = self.get_default_bond_allowable_set(bond_characteristic)
            if not isinstance(allowable_set, (list, tuple)):
                raise TypeError("Input dictionary must have values of type list, tuple or None.")
            self.bond_allowable_sets[bond_characteristic] = list(allowable_set)
    

    def get_atom_feature_labels(self):
        """
        Returns the labels for the atom features.

        Returns:
            list: The labels for the atom features.
        """
        
        atom_feature_labels = []
        
        for characteristic in self.atom_allowable_sets.keys():
            if characteristic in self.NON_ONE_HOT_ENCODED:
                atom_feature_labels.append(characteristic)
            else:
                atom_feature_labels += [f"{characteristic}_{value}" for value in self.atom_allowable_sets[characteristic]]
        
        return atom_feature_labels
    
    
    def get_bond_feature_labels(self):
        """
        Returns the labels for the bond features.

        Returns:
            list: The labels for the bond features.
        """
        
        bond_feature_labels = []
        
        for characteristic in self.bond_allowable_sets.keys():
            if characteristic in self.NON_ONE_HOT_ENCODED:
                bond_feature_labels.append(characteristic)
            else:
                bond_feature_labels += [f"{characteristic}_{value}" for value in self.bond_allowable_sets[characteristic]]
        
        return bond_feature_labels
    

    def get_default_atom_allowable_set(self, atom_characteristic):
        """
        Returns the default allowable set for an atom characteristic.

        Args:
            atom_characteristic (str): The atom characteristic.

        Returns:
            list: The default allowable set for the atom characteristic.
        """

        match atom_characteristic:
            case "symbol":
                return ['C', 'O', 'N', 'Cl', 'S', 'F', 'Na', 'P', 'Br', 'Si', 'K', 'Sn', 'UNK'],
            case "degree":
                return [2, 1, 3, 4, 0]
            case "hybridization":
                return [Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3,
                        Chem.rdchem.HybridizationType.S,
                        Chem.rdchem.HybridizationType.SP]
            case "total_num_hs":
                return [0, 1, 2, 3]
            case "implicit_valence":
                return [0, 1, 2, 3, 4]
            case "explicit_valence":
                return [4, 2, 3, 1, 0, 'UNK']
            case "formal_charge":
                return [0, -1, 1, 'UNK']
            case _:
                raise ValueError(f"No default allowable set for atom characteristic '{atom_characteristic}'. You must set your own allowable set.")

    
    def get_default_bond_allowable_set(self, bond_characteristic):
        """
        Returns the default allowable set for a bond characteristic.

        Args:
            bond_characteristic (str): The bond characteristic.

        Returns:
            list: The default allowable set for the bond characteristic.
        """

        match bond_characteristic:
            case "bond_type":
                return [Chem.rdchem.BondType.SINGLE,
                        Chem.rdchem.BondType.DOUBLE,
                        Chem.rdchem.BondType.TRIPLE,
                        Chem.rdchem.BondType.AROMATIC]
            case _:
                raise ValueError(f"No default allowable set for bond characteristic '{bond_characteristic}'. You must set your own allowable set.")
        

    def set_default_config(self):
        """
        Sets the default configuration for the featurizer.
        """

        atom_allowable_sets = {
            "symbol": self.get_default_atom_allowable_set("symbol"),
            "degree": self.get_default_atom_allowable_set("degree"),
            "hybridization": self.get_default_atom_allowable_set("hybridization"),
            "total_num_hs": self.get_default_atom_allowable_set("total_num_hs"),
            "implicit_valence": self.get_default_atom_allowable_set("implicit_valence"),
            "explicit_valence": self.get_default_atom_allowable_set("explicit_valence"),
            "is_aromatic": None,
            "_ChiralityPossible": None,
            "formal_charge": self.get_default_atom_allowable_set("formal_charge"),
        }

        bond_allowable_sets = {
            "bond_type": self.get_default_bond_allowable_set("bond_type"),
            "is_conjugated": None,
            "is_in_ring": None,
        }


        self.set_atom_allowable_sets(atom_allowable_sets)

        if self.include_edge_features:
            self.set_bond_allowable_sets(bond_allowable_sets)
    
    
    def extract_molecular_features(self, mol):
        """
        Extracts molecular features from a molecule.

        Args:
            mol (rdkit.Chem.Mol): The molecule to extract features from.

        Returns:
            tuple: The atom features and bond features as tensors.
        """
        
        mol_atom_features = []
        for atom in mol.GetAtoms():
            mol_atom_features.append(self.atom_features(atom))
        mol_atom_features = torch.tensor(mol_atom_features, dtype=torch.float32)
        
        if not self.include_edge_features:
            return mol_atom_features, None
        
        mol_bond_features = []
        for bond in mol.GetBonds():
            mol_bond_features.append(self.bond_features(bond))
            mol_bond_features.append(self.bond_features(bond)) # do twice (undirectional graph)
        mol_bond_features = torch.tensor(mol_bond_features, dtype=torch.float32)
        
        return mol_atom_features, mol_bond_features
        
        
    def atom_features(self, atom):
        """
        Extracts features for an atom.

        Args:
            atom (rdkit.Chem.Atom): The atom to extract features from.

        Returns:
            list: The features for the atom.
        """
        
        feats = []
        for characteristic in self.atom_allowable_sets.keys():
            property_getter = self.SUPPORTED_ATOM_CHARACTERISTICS[characteristic]
            feat = property_getter(atom)
            
            if characteristic in self.NON_ONE_HOT_ENCODED:
                feats.append(feat)
            else:
                allowable_set = self.atom_allowable_sets[characteristic]
                if 'UNK' in allowable_set:
                    one_hot_encoded_feat = self.one_of_k_encoding_unk(feat, allowable_set)
                else:
                    one_hot_encoded_feat = self.one_of_k_encoding(feat, allowable_set, characteristic_name=characteristic)
                feats.extend(one_hot_encoded_feat)
        return feats
    

    def bond_features(self, bond):
        """
        Extracts features for a bond.

        Args:
            bond (rdkit.Chem.Bond): The bond to extract features from.

        Returns:
            list: The features for the bond.
        """

        feats = []
        for characteristic in self.bond_allowable_sets.keys():
            property_getter = self.SUPPORTED_BOND_CHARACTERISTICS[characteristic]
            feat = property_getter(bond)
            
            if characteristic in self.NON_ONE_HOT_ENCODED:
                feats.append(feat)
            else:
                allowable_set = self.bond_allowable_sets[characteristic]
                if 'UNK' in allowable_set:
                    one_hot_encoded_feat = self.one_of_k_encoding_unk(feat, allowable_set)
                else:
                    one_hot_encoded_feat = self.one_of_k_encoding(feat, allowable_set, characteristic_name=characteristic)
                feats.extend(one_hot_encoded_feat)
        return feats

                
    def one_of_k_encoding(self, x, allowable_set, characteristic_name=None):
        """
        One-hot encodes a value based on an allowable set.

        Args:
            x: The value to encode.
            allowable_set (list): The allowable set for the value.
            characteristic_name (str): The name of the characteristic (optional).

        Returns:
            list: The one-hot encoded vector.
        """
        if x not in allowable_set:
            characteristic_text = f"{characteristic_name} " if characteristic_name else ""
            self.warning(f"Ignoring input {characteristic_text}{x}, not in allowable set {allowable_set}")
        return [x == s for s in allowable_set]


    def one_of_k_encoding_unk(self, x, allowable_set):
        """
        One-hot encodes a value based on an allowable set, mapping unknown values to 'UNK'.

        Args:
            x: The value to encode.
            allowable_set (list): The allowable set for the value.

        Returns:
            list: The one-hot encoded value.
        """
        if x not in allowable_set:
            x = 'UNK'
        return [x == s for s in allowable_set]
    
    
    def adjacency_matrix(self, mol):
        """
        Computes the adjacency matrix for a molecule.

        Args:
            mol (rdkit.Chem.Mol): The molecule to compute the adjacency matrix for.

        Returns:
            torch.Tensor: The adjacency matrix.
        """
    
        ix1, ix2 = [], []
        for bond in mol.GetBonds():

            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ix1 += [start, end]
            ix2 += [end, start]

        adj_norm = torch.asarray([ix1, ix2], dtype=torch.int64) # Needs to be in COO Format
        return adj_norm
        
        
    def get_num_node_features(self):
        """
        Returns the number of node features.

        Returns:
            int: The number of node features.
        """
        return len(self.get_atom_feature_labels())
    

    def get_num_edge_features(self):
        """
        Returns the number of edge features.

        Returns:
            int: The number of edge features.
        """
        return len(self.get_bond_feature_labels())
    
    
    def warning(self, message):
        """
        Issues a warning if warnings are enabled.

        Args:
            message (str): The warning message.
        """
        if self.warnings_enabled:
            warnings.warn(message)
    
    def enable_warnings(self, enable=True):
        """
        Enables or disables warnings.

        Args:
            enable (bool): Whether to enable warnings.
        """
        self.warnings_enabled = enable
    
    
    def featurize(self, sm, y=None):
        """
        Featurizes a SMILES string into graph data.

        Args:
            sm (str): The SMILES string.
            y: The target value (optional).

        Returns:
            torch_geometric.data.Data: The featurized data.
        """
        
        mol = Chem.MolFromSmiles(sm)
        adjacency_matrix = self.adjacency_matrix(mol)
        mol_atom_features, mol_bond_features = self.extract_molecular_features(mol)
        
        return Data(x=mol_atom_features,
                    edge_index=adjacency_matrix,
                    edge_attr=mol_bond_features,
                    y=y,
                    smiles=sm)
        
    
    def save_config(self, config_file="featurizer_config.pkl"):
        """
        Saves the featurizer configuration to a file.

        Args:
            config_file (str): The file to save the configuration to.
        """
        config = {
            'warnings_enabled': self.warnings_enabled,
            'include_edge_features': self.include_edge_features,
            'atom_allowable_sets': self.atom_allowable_sets,
            'bond_allowable_sets': self.bond_allowable_sets,
        }

        with open(config_file, 'wb') as f:
            pickle.dump(config, f)

        
    def load_config(self, config_file):
        """
        Loads the featurizer configuration from a file.

        Args:
            config_file (str): The file to load the configuration from.

        Returns:
            SmilesGraphFeaturizer: The configured featurizer.
        """

        with open(config_file, 'rb') as f:
            config = pickle.load(f)

        self.warnings_enabled = config['warnings_enabled']
        self.include_edge_features = config['include_edge_features']

        self.set_atom_allowable_sets(config['atom_allowable_sets'])

        self.set_bond_allowable_sets(config['bond_allowable_sets'])

        return self
    

    def save(self, filepath='featurizer.pkl'):
        """
        Saves the featurizer to a file.

        Args:
            filepath (str): The file to save the featurizer to.
        """
        with open(filepath, "wb") as file:
            pickle.dump(self, file)
    

    def __call__(self, sm, y=None):
        """
        Featurizes a SMILES string into graph data.

        Args:
            sm (str): The SMILES string.
            y: The target value (optional).

        Returns:
            torch_geometric.data.Data: The featurized data.
        """
        return self.featurize(sm, y)
    

    def __repr__(self):
        """
        Returns the string representation of the featurizer.

        Returns:
            str: The string representation of the featurizer.
        """
        attributes = {
            'atom_allowable_sets': self.atom_allowable_sets,
            'bond_allowable_sets': self.bond_allowable_sets,
        }
        return type(self).__name__ + '(' + json.dumps(attributes, indent=4) + ')'
    
    def __str__(self):
        """
        Returns the string representation of the featurizer.

        Returns:
            str: The string representation of the featurizer.
        """
        return self.__repr__()