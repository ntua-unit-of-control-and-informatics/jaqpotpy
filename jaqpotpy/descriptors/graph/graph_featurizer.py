from rdkit import Chem
import torch
from torch_geometric.data import Data
import json
from collections import OrderedDict


class SmilesGraphFeaturizer:
    """
    Featurizes SMILES strings into graph data suitable for graph neural networks.
    Class Attributes:
    SUPPORTED_ATOM_FEATURES (dict): Dictionary mapping atom features to their respective featurization function.
    SUPPORTED_BOND_FEATURES (dict): Dictionary mapping bond features to their respective featurization function.
    NON_ONE_HOT_ENCODED (list): List of features that should not be one-hot encoded.
    """

    SUPPORTED_ATOM_FEATURES = {
        "symbol": lambda atom: atom.GetSymbol(),
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
        "_ChiralityPossible": lambda atom: atom.HasProp("_ChiralityPossible"),
        "isotope": lambda atom: atom.GetIsotope(),
        "total_valence": lambda atom: atom.GetTotalValence(),
        "explicit_valence": lambda atom: atom.GetExplicitValence(),
        "implicit_valence": lambda atom: atom.GetImplicitValence(),
        "chiral_tag": lambda atom: atom.GetChiralTag(),
        "mass": lambda atom: (atom.GetMass() - 14.5275) / 9.4154,
    }
    SUPPORTED_BOND_FEATURES = {
        "bond_type": lambda bond: bond.GetBondType(),
        "is_conjugated": lambda bond: bond.GetIsConjugated(),
        "is_in_ring": lambda bond: bond.IsInRing(),
        "stereo": lambda bond: bond.GetStereo(),
    }
    NON_ONE_HOT_ENCODED = [
        "formal_charge",
        "num_radical_electrons",
        "is_aromatic",
        "_ChiralityPossible",
        "is_conjugated",
        "is_in_ring",
        "mass",
    ]

    def __init__(self, include_edge_features=True):
        """include_edge_features (bool): Whether to include edge features in the featurization."""
        self.include_edge_features = include_edge_features
        self.atom_allowable_sets = {}
        self.bond_allowable_sets = {}

    def _set_atom_allowable_sets(self, atom_allowable_sets_dict):
        """
        Sets the allowable sets for atom feature. This is used internally.
        Atom_allowable_sets_dict (dict): Dictionary of allowable sets per atom feature.
        """
        for atom_feature, allowable_set in atom_allowable_sets_dict.items():
            self.add_atom_feature(atom_feature, allowable_set)

    def _set_bond_allowable_sets(self, bond_allowable_sets_dict):
        """
        Sets the allowable sets for bond features. This is used internally.
        Bond_allowable_sets_dict (dict): Dictionary of allowable sets per bond feature.
        """
        for bond_feature, allowable_set in bond_allowable_sets_dict.items():
            self.add_bond_feature(bond_feature, allowable_set)

    def add_atom_feature(self, atom_feature, allowable_set=None):
        """
        Adds an atom feature to the featurizer. Used externally to add atom features.
        -atom_feature (str): The atom feature name to be added.
        -allowable_set (list or None): The allowable set for the atom feature.
        'UNK' is used as a special keyword. When an atom feature is encoded and an unknown
        value (not in the allowable set) is encountered during featurization, it will be encoded as 'UNK'.
        This allows handling of unexpected values by mapping them to a default or placeholder category.
        """
        # Error handling when user input feature name does not match self.SUPPORTED_ATOM_FEATURES
        if atom_feature not in self.SUPPORTED_ATOM_FEATURES.keys():
            raise ValueError(f"Unsupported atom feature '{atom_feature}'")
        # Placeholder for the feature to contain a single element for non one hot encoded integers
        if atom_feature in self.NON_ONE_HOT_ENCODED:
            self.atom_allowable_sets[atom_feature] = None
        else:
            # If user does not provide allowable set, use default
            if allowable_set is None:
                allowable_set = self.get_default_atom_allowable_set(atom_feature)
            # Error handling for user input allowable set data type
            elif not isinstance(allowable_set, (list, tuple)):
                raise TypeError(
                    "Input dictionary must have values of type list, tuple or None."
                )
            # Creates a dict where key is the string name of the feature and values are the allowable set
            # This is if user input provides allowable set based on RDKit
            else:
                self.atom_allowable_sets[atom_feature] = list(allowable_set)

    def add_bond_feature(self, bond_feature, allowable_set=None):
        """
        Adds a bond feature to the featurizer. Used externally to add bond features.
        -bond_feature (str): The bond feature to add.
        -allowable_set (list or None): The allowable set for the bond feature.
        'UNK' is used as a special keyword. When a bond feature is encoded and an unknown
        value (not in the allowable set) is encountered during featurization, it will be encoded as 'UNK'.
        This allows handling of unexpected values by mapping them to a default or placeholder category.
        """
        # Error handling when user input feature name does not match self.SUPPORTED_ATOM_FEATURES
        if bond_feature not in self.SUPPORTED_BOND_FEATURES.keys():
            raise ValueError(f"Unsupported bond feature '{bond_feature}'")
        if bond_feature in self.NON_ONE_HOT_ENCODED:
            # Placeholder for the feature to contain a single element for non one hot encoded integers
            self.bond_allowable_sets[bond_feature] = None
        else:
            # If user does not provide allowable set, use default
            if allowable_set is None:
                allowable_set = self.get_default_bond_allowable_set(bond_feature)
            # Error handling for user input allowable set data type
            elif not isinstance(allowable_set, (list, tuple)):
                raise TypeError(
                    "Input dictionary must have values of type list, tuple or None."
                )
            # Creates a dict where key is the string name of the feature and values are the allowable set
            # This is if user input provides allowable set based on RDKit
            else:
                self.bond_allowable_sets[bond_feature] = list(allowable_set)

    def get_default_atom_allowable_set(self, atom_feature):
        """Returns the default allowable set for an atom feature."""
        match atom_feature:
            case "symbol":
                return [
                    "C",
                    "O",
                    "N",
                    "Cl",
                    "S",
                    "F",
                    "Na",
                    "P",
                    "Br",
                    "Si",
                    "K",
                    "Sn",
                    "UNK",
                ]
            case "degree":
                return [0, 1, 2, 3, 4]
            case "hybridization":
                return [
                    Chem.rdchem.HybridizationType.S,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2,
                ]
            case "total_num_hs":
                return [0, 1, 2, 3, 4]
            case "implicit_valence":
                return [0, 1, 2, 3, 4]
            case "explicit_valence":
                return [4, 2, 3, 1, 0, "UNK"]
            case _:
                raise ValueError(
                    f"No default allowable set for atom characteristic '{atom_feature}'. You must set your own allowable set."
                )

    def get_default_bond_allowable_set(self, bond_feature):
        """Returns the default allowable set for a bond feature."""
        match bond_feature:
            case "bond_type":
                return [
                    Chem.rdchem.BondType.SINGLE,
                    Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE,
                    Chem.rdchem.BondType.AROMATIC,
                ]
            case _:
                raise ValueError(
                    f"No default allowable set for bond characteristic '{bond_feature}'. You must set your own allowable set."
                )

    def sort_allowable_sets(self):
        """Sorts the allowable sets for atom and bond features."""
        self.atom_allowable_sets = OrderedDict(sorted(self.atom_allowable_sets.items()))
        self.bond_allowable_sets = OrderedDict(sorted(self.bond_allowable_sets.items()))

        return self

    def set_default_config(self):
        """Sets the default configuration for the featurizer."""
        atom_allowable_sets = {
            "symbol": self.get_default_atom_allowable_set("symbol"),
            "degree": self.get_default_atom_allowable_set("degree"),
            "hybridization": self.get_default_atom_allowable_set("hybridization"),
            "total_num_hs": self.get_default_atom_allowable_set("total_num_hs"),
            "is_aromatic": None,
            "formal_charge": None,
        }

        bond_allowable_sets = {
            "bond_type": self.get_default_bond_allowable_set("bond_type"),
            "is_conjugated": None,
            "is_in_ring": None,
        }
        self._set_atom_allowable_sets(atom_allowable_sets)
        if self.include_edge_features:
            self._set_bond_allowable_sets(bond_allowable_sets)

    def _extract_molecular_features(self, mol):
        """Extracts molecular features from a molecule. Returns the atom and bond (optionally) features for a molecule."""
        mol_atom_features = []
        for atom in mol.GetAtoms():
            mol_atom_features.append(self._atom_features(atom))
        mol_atom_features = torch.tensor(mol_atom_features, dtype=torch.float32)
        if not self.include_edge_features:
            return mol_atom_features, None
        mol_bond_features = []
        for bond in mol.GetBonds():
            mol_bond_features.append(self._bond_features(bond))
            mol_bond_features.append(self._bond_features(bond))
        mol_bond_features = torch.tensor(mol_bond_features, dtype=torch.float32)
        return mol_atom_features, mol_bond_features

    def _atom_features(self, atom):
        """Extracts features for an atom. Used internally."""
        feats = []
        for feature in self.atom_allowable_sets.keys():
            # Contains the rdkit function for feature calculation
            property_getter = self.SUPPORTED_ATOM_FEATURES[feature]
            feat = property_getter(atom)
            if feature in self.NON_ONE_HOT_ENCODED:
                feats.append(feat)
            else:
                allowable_set = self.atom_allowable_sets[feature]
                if "UNK" in allowable_set:
                    one_hot_encoded_feat = self._one_of_k_encoding_unk(
                        feat, allowable_set
                    )
                else:
                    one_hot_encoded_feat = self._one_of_k_encoding(
                        feat, allowable_set, feature_name=feature
                    )
                feats.extend(one_hot_encoded_feat)
        return feats

    def _bond_features(self, bond):
        """Extracts features for a bond. Used internally."""
        feats = []
        for feature in self.bond_allowable_sets.keys():
            property_getter = self.SUPPORTED_BOND_FEATURES[feature]
            feat = property_getter(bond)

            if feature in self.NON_ONE_HOT_ENCODED:
                feats.append(feat)
            else:
                allowable_set = self.bond_allowable_sets[feature]
                if "UNK" in allowable_set:
                    one_hot_encoded_feat = self._one_of_k_encoding_unk(
                        feat, allowable_set
                    )
                else:
                    one_hot_encoded_feat = self._one_of_k_encoding(
                        feat, allowable_set, feature_name=feature
                    )
                feats.extend(one_hot_encoded_feat)
        return feats

    def _one_of_k_encoding(self, x, allowable_set, feature_name=None):
        """One-hot encodes a value based on an allowable set."""
        if x not in allowable_set:
            feature_text = f"{feature_name} " if feature_name else ""
            print(feature_text)
        return [x == s for s in allowable_set]

    def _one_of_k_encoding_unk(self, x, allowable_set):
        """One-hot encodes a value based on an allowable set, mapping unseen values to 'UNK'."""
        if x not in allowable_set:
            x = "UNK"
        return [x == s for s in allowable_set]

    def _adjacency_matrix(self, mol):
        """Computes the adjacency matrix for a molecule. Used internally"""
        ix1, ix2 = [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ix1 += [start, end]
            ix2 += [end, start]
        adj_norm = torch.asarray(
            [ix1, ix2], dtype=torch.int64
        )  # Needs to be in COO Format
        return adj_norm

    def get_num_node_features(self):
        """Returns the number of node features."""
        num = 0
        for value in self.atom_allowable_sets.values():
            if value is None:
                num += 1
            else:
                num += len(value)
        return num

    def get_num_edge_features(self):
        """Returns the number of edge features"""
        num = 0
        for value in self.bond_allowable_sets.values():
            if value is None:
                num += 1
            else:
                num += len(value)
        return num

    def __call__(self, *args, **kwargs):
        """Featurizes the input data"""
        return self.featurize(*args, **kwargs)

    def featurize(self, sm, y=None):
        """Featurizes a SMILES string into graph data. Returns torch_geometric.data.Data object."""
        mol = Chem.MolFromSmiles(sm)
        adjacency_matrix = self._adjacency_matrix(mol)
        mol_atom_features, mol_bond_features = self._extract_molecular_features(mol)
        return Data(
            x=mol_atom_features,
            edge_index=adjacency_matrix,
            edge_attr=mol_bond_features,
            y=y,
            smiles=sm,
        )

    def get_dict(self):
        """Creates a json configuration that will be sent to database for inference"""
        config_dict = {
            "include_edge_features": self.include_edge_features,
            "atom_allowable_sets": self.atom_allowable_sets,
            "bond_allowable_sets": self.bond_allowable_sets,
        }
        return config_dict

    def load_dict(self, feat_dict):
        """Loads a configuration from a json dict. Mainly used in jaqpotpy-inference"""
        self.include_edge_features = feat_dict["include_edge_features"]
        self._set_atom_allowable_sets(feat_dict["atom_allowable_sets"])
        self._set_bond_allowable_sets(feat_dict["bond_allowable_sets"])
        return self

    def __repr__(self):
        """Representation of the featurizer"""
        attributes = {
            "atom_allowable_sets": self.atom_allowable_sets,
            "bond_allowable_sets": self.bond_allowable_sets,
        }
        return type(self).__name__ + "(" + json.dumps(attributes, indent=4) + ")"

    def __str__(self):
        """Returns the string representation of the featurizer."""
        return self.__repr__()
