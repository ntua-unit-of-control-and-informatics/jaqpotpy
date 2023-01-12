from typing import List, Tuple
import numpy as np
import itertools
from functools import partial

from jaqpotpy.utils.types import RDKitAtom, RDKitBond, RDKitMol
from jaqpotpy.descriptors.graph.graph_data import GraphData
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from jaqpotpy.utils.molecule_feature_utils import one_hot_encode
from jaqpotpy.utils.molecule_feature_utils import get_atom_type_one_hot
from jaqpotpy.utils.molecule_feature_utils import get_atom_num_radical_electrons
from jaqpotpy.utils.molecule_feature_utils import construct_hydrogen_bonding_info
from jaqpotpy.utils.molecule_feature_utils import get_atom_hydrogen_bonding_one_hot
from jaqpotpy.utils.molecule_feature_utils import get_atom_hybridization_one_hot
from jaqpotpy.utils.molecule_feature_utils import get_atom_is_aromatic
from jaqpotpy.utils.molecule_feature_utils import get_atom_total_num_Hs_one_hot
from jaqpotpy.utils.molecule_feature_utils import get_atom_is_in_aromatic_one_hot
from jaqpotpy.utils.molecule_feature_utils import get_atom_chirality_one_hot
from jaqpotpy.utils.molecule_feature_utils import get_atom_formal_charge
from jaqpotpy.utils.molecule_feature_utils import get_atom_partial_charge
from jaqpotpy.utils.molecule_feature_utils import get_atom_total_degree_one_hot
from jaqpotpy.utils.molecule_feature_utils import get_atom_degree_one_hot
from jaqpotpy.utils.molecule_feature_utils import get_bond_type_one_hot
from jaqpotpy.utils.molecule_feature_utils import get_bond_is_in_same_ring_one_hot
from jaqpotpy.utils.molecule_feature_utils import get_bond_is_conjugated_one_hot
from jaqpotpy.utils.molecule_feature_utils import get_bond_stereo_one_hot
from jaqpotpy.utils.molecule_feature_utils import get_atom_formal_charge_one_hot
from jaqpotpy.utils.molecule_feature_utils import get_atom_implicit_valence_one_hot
from jaqpotpy.utils.molecule_feature_utils import get_atom_explicit_valence_one_hot
from jaqpotpy.utils.molecule_feature_utils import get_atom_is_chiral_center
from jaqpotpy.utils.rdkit_utils import compute_all_pairs_shortest_path
from jaqpotpy.utils.rdkit_utils import compute_pairwise_ring_info


def _construct_atom_feature(
        atom: RDKitAtom, h_bond_infos: List[Tuple[int, str]], use_chirality: bool,
        use_partial_charge: bool) -> np.ndarray:
    """Construct an atom feature from a RDKit atom object.
  Parameters
  ----------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object
  h_bond_infos: List[Tuple[int, str]]
    A list of tuple `(atom_index, hydrogen_bonding_type)`.
    Basically, it is expected that this value is the return value of
    `construct_hydrogen_bonding_info`. The `hydrogen_bonding_type`
    value is "Acceptor" or "Donor".
  use_chirality: bool
    Whether to use chirality information or not.
  use_partial_charge: bool
    Whether to use partial charge data or not.
  Returns
  -------
  np.ndarray
    A one-hot vector of the atom feature.
  """
    atom_type = get_atom_type_one_hot(atom)
    formal_charge = get_atom_formal_charge(atom)
    hybridization = get_atom_hybridization_one_hot(atom)
    acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
    aromatic = get_atom_is_in_aromatic_one_hot(atom)
    degree = get_atom_total_degree_one_hot(atom)
    total_num_Hs = get_atom_total_num_Hs_one_hot(atom)
    atom_feat = np.concatenate([
        atom_type, formal_charge, hybridization, acceptor_donor, aromatic, degree,
        total_num_Hs
    ])

    if use_chirality:
        chirality = get_atom_chirality_one_hot(atom)
        atom_feat = np.concatenate([atom_feat, chirality])

    if use_partial_charge:
        partial_charge = get_atom_partial_charge(atom)
        atom_feat = np.concatenate([atom_feat, partial_charge])
    return atom_feat


def _construct_bond_feature(bond: RDKitBond) -> np.ndarray:
    """Construct a bond feature from a RDKit bond object.
  Parameters
  ---------
  bond: rdkit.Chem.rdchem.Bond
    RDKit bond object
  Returns
  -------
  np.ndarray
    A one-hot vector of the bond feature.
  """
    bond_type = get_bond_type_one_hot(bond)
    same_ring = get_bond_is_in_same_ring_one_hot(bond)
    conjugated = get_bond_is_conjugated_one_hot(bond)
    stereo = get_bond_stereo_one_hot(bond)
    return np.concatenate([bond_type, same_ring, conjugated, stereo])


class MolGraphConvFeaturizer(MolecularFeaturizer):
    """This class is a featurizer of general graph convolution networks for molecules.
      The default node(atom) and edge(bond) representations are based on
      `WeaveNet paper <https://arxiv.org/abs/1603.00856>`_. If you want to use your own representations,
      you could use this class as a guide to define your original Featurizer. In many cases, it's enough
      to modify return values of `construct_atom_feature` or `construct_bond_feature`.
      The default node representation are constructed by concatenating the following values,
      and the feature length is 30.
      - Atom type: A one-hot vector of this atom, "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other atoms".
      - Formal charge: Integer electronic charge.
      - Hybridization: A one-hot vector of "sp", "sp2", "sp3".
      - Hydrogen bonding: A one-hot vector of whether this atom is a hydrogen bond donor or acceptor.
      - Aromatic: A one-hot vector of whether the atom belongs to an aromatic ring.
      - Degree: A one-hot vector of the degree (0-5) of this atom.
      - Number of Hydrogens: A one-hot vector of the number of hydrogens (0-4) that this atom connected.
      - Chirality: A one-hot vector of the chirality, "R" or "S". (Optional)
      - Partial charge: Calculated partial charge. (Optional)
      The default edge representation are constructed by concatenating the following values,
      and the feature length is 11.
      - Bond type: A one-hot vector of the bond type, "single", "double", "triple", or "aromatic".
      - Same ring: A one-hot vector of whether the atoms in the pair are in the same ring.
      - Conjugated: A one-hot vector of whether this bond is conjugated or not.
      - Stereo: A one-hot vector of the stereo configuration of a bond.
      If you want to know more details about features, please check the paper [1]_ and
      utilities in deepchem.utils.molecule_feature_utils.py.
      Examples
      --------
      >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
      >>> featurizer = MolGraphConvFeaturizer(use_edges=True)
      >>> out = featurizer.featurize(smiles)
      >>> type(out[0])
      <class 'jaqpotpy.descriptors.graph.GraphData'>
      >>> out[0].num_node_features
      30
      >>> out[0].num_edge_features
      11
      References
      ----------
      .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints."
         Journal of computer-aided molecular design 30.8 (2016):595-608.
      Note
      ----
      This class requires RDKit to be installed.
      """

    @property
    def __name__(self):
        return 'MolGraphConvFeaturizer'

    def __init__(self,
                 use_edges: bool = False,
                 use_chirality: bool = False,
                 use_partial_charge: bool = False):
        """
    Parameters
    ----------
    use_edges: bool, default False
      Whether to use edge features or not.
    use_chirality: bool, default False
      Whether to use chirality information or not.
      If True, featurization becomes slow.
    use_partial_charge: bool, default False
      Whether to use partial charge data or not.
      If True, this featurizer computes gasteiger charges.
      Therefore, there is a possibility to fail to featurize for some molecules
      and featurization becomes slow.
    """
        self.use_edges = use_edges
        self.use_partial_charge = use_partial_charge
        self.use_chirality = use_chirality

    def __getitem__(self):
        return self

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        """Calculate molecule graph features from RDKit mol object.
    Parameters
    ----------
    datapoint: rdkit.Chem.rdchem.Mol
      RDKit mol object.
    Returns
    -------
    graph: GraphData
      A molecule graph with some features.
    """
        assert datapoint.GetNumAtoms(
        ) > 1, "More than one atom should be present in the molecule for this featurizer to work."
        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )

        if self.use_partial_charge:
            try:
                datapoint.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
            except:
                # If partial charges were not computed
                try:
                    from rdkit.Chem import AllChem
                    AllChem.ComputeGasteigerCharges(datapoint)
                except ModuleNotFoundError:
                    raise ImportError("This class requires RDKit to be installed.")

        # construct atom (node) feature
        h_bond_infos = construct_hydrogen_bonding_info(datapoint)
        atom_features = np.asarray(
            [
                _construct_atom_feature(atom, h_bond_infos, self.use_chirality,
                                        self.use_partial_charge)
                for atom in datapoint.GetAtoms()
            ],
            dtype=float,
        )

        # construct edge (bond) index
        src, dest = [], []
        for bond in datapoint.GetBonds():
            # add edge list considering a directed graph
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]

        # construct edge (bond) feature
        bond_features = None  # deafult None
        if self.use_edges:
            features = []
            for bond in datapoint.GetBonds():
                features += 2 * [_construct_bond_feature(bond)]
            bond_features = np.asarray(features, dtype=float)

        return GraphData(
            node_features=atom_features,
            edge_index=np.asarray([src, dest], dtype=int),
            edge_features=bond_features)

    def _get_column_names(self, **kwargs) -> list:
        """
        Return the column names
        """
        names = ['MoleculeGraph']
        return names

    def _featurize_dataframe(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        """Calculate molecule graph features from RDKit mol object.
        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
          RDKit mol object.
        Returns
        -------
        graph: GraphData
          A molecule graph with some features.
        """
        assert datapoint.GetNumAtoms(
        ) > 1, "More than one atom should be present in the molecule for this featurizer to work."
        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )

        if self.use_partial_charge:
            try:
                datapoint.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
            except:
                # If partial charges were not computed
                try:
                    from rdkit.Chem import AllChem
                    AllChem.ComputeGasteigerCharges(datapoint)
                except ModuleNotFoundError:
                    raise ImportError("This class requires RDKit to be installed.")

        # construct atom (node) feature
        h_bond_infos = construct_hydrogen_bonding_info(datapoint)
        atom_features = np.asarray(
            [
                _construct_atom_feature(atom, h_bond_infos, self.use_chirality,
                                        self.use_partial_charge)
                for atom in datapoint.GetAtoms()
            ],
            dtype=float,
        )

        # construct edge (bond) index
        src, dest = [], []
        for bond in datapoint.GetBonds():
            # add edge list considering a directed graph
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]

        # construct edge (bond) feature
        bond_features = None  # deafult None
        if self.use_edges:
            features = []
            for bond in datapoint.GetBonds():
                features += 2 * [_construct_bond_feature(bond)]
            bond_features = np.asarray(features, dtype=float)

        return GraphData(
            node_features=atom_features,
            edge_index=np.asarray([src, dest], dtype=int),
            edge_features=bond_features)


class PagtnMolGraphFeaturizer(MolecularFeaturizer):
    """This class is a featuriser of PAGTN graph networks for molecules.
  The featurization is based on `PAGTN model <https://arxiv.org/abs/1905.12712>`_. It is
  slightly more computationally intensive than default Graph Convolution Featuriser, but it
  builds a Molecular Graph connecting all atom pairs accounting for interactions of an atom with
  every other atom in the Molecule. According to the paper, interactions between two pairs
  of atom are dependent on the relative distance between them and and hence, the function needs
  to calculate the shortest path between them.
  The default node representation is constructed by concatenating the following values,
  and the feature length is 94.
  - Atom type: One hot encoding of the atom type. It consists of the most possible elements in a chemical compound.
  - Formal charge: One hot encoding of formal charge of the atom.
  - Degree: One hot encoding of the atom degree
  - Explicit Valence: One hot encoding of explicit valence of an atom. The supported possibilities
    include ``0 - 6``.
  - Implicit Valence: One hot encoding of implicit valence of an atom. The supported possibilities
    include ``0 - 5``.
  - Aromaticity: Boolean representing if an atom is aromatic.
  The default edge representation is constructed by concatenating the following values,
  and the feature length is 42. It builds a complete graph where each node is connected to
  every other node. The edge representations are calculated based on the shortest path between two nodes
  (choose any one if multiple exist). Each bond encountered in the shortest path is used to
  calculate edge features.
  - Bond type: A one-hot vector of the bond type, "single", "double", "triple", or "aromatic".
  - Conjugated: A one-hot vector of whether this bond is conjugated or not.
  - Same ring: A one-hot vector of whether the atoms in the pair are in the same ring.
  - Ring Size and Aromaticity: One hot encoding of atoms in pair based on ring size and aromaticity.
  - Distance: One hot encoding of the distance between pair of atoms.
  Examples
  --------
  >>> from jaqpotpy.descriptors.molecular import PagtnMolGraphFeaturizer
  >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
  >>> featurizer = PagtnMolGraphFeaturizer(max_length=5)
  >>> out = featurizer.featurize(smiles)
  >>> type(out[0])
  <class 'deepchem.feat.graph_data.GraphData'>
  >>> out[0].num_node_features
  94
  >>> out[0].num_edge_features
  42
  References
  ----------
  .. [1] Chen, Barzilay, Jaakkola "Path-Augmented Graph Transformer Network"
     10.26434/chemrxiv.8214422.
  Note
  ----
  This class requires RDKit to be installed.
  """

    @property
    def __name__(self):
        return 'PagtnMolGraphFeaturizer'

    def __init__(self, max_length=5):
        """
      Parameters
      ----------
      max_length : int
        Maximum distance up to which shortest paths must be considered.
        Paths shorter than max_length will be padded and longer will be
        truncated, default to ``5``.
      """

        self.SYMBOLS = [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
            'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd',
            'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
            'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh',
            'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga',
            'Cs', '*', 'UNK'
        ]

        self.RING_TYPES = [(5, False), (5, True), (6, False), (6, True)]
        self.ordered_pair = lambda a, b: (a, b) if a < b else (b, a)
        self.max_length = max_length

    def __getitem__(self):
        return self

    def _get_column_names(self, **kwargs) -> list:
        """
        Return the column names
        """
        names = ['PagtnMolGraphFeaturizer']
        return names

    def _pagtn_atom_featurizer(self, atom: RDKitAtom) -> np.ndarray:
        """Calculate Atom features from RDKit atom object.
    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
      RDKit mol object.
    Returns
    -------
    atom_feat: np.ndarray
      numpy vector of atom features.
    """
        atom_type = get_atom_type_one_hot(atom, self.SYMBOLS, False)
        formal_charge = get_atom_formal_charge_one_hot(
            atom, include_unknown_set=False)
        degree = get_atom_total_degree_one_hot(atom, list(range(11)), False)
        exp_valence = get_atom_explicit_valence_one_hot(atom, list(range(7)), False)
        imp_valence = get_atom_implicit_valence_one_hot(atom, list(range(6)), False)
        armoticity = get_atom_is_in_aromatic_one_hot(atom)
        atom_feat = np.concatenate([
            atom_type, formal_charge, degree, exp_valence, imp_valence, armoticity
        ])
        return atom_feat

    def _edge_features(self, mol: RDKitMol, path_atoms: Tuple[int, ...],
                       ring_info) -> np.ndarray:
        """Computes the edge features for a given pair of nodes.
        Parameters
        ----------
        mol : : RDKitMol
            RDKit molecule instance.
        path_atoms: tuple
            Shortest path between the given pair of nodes.
        ring_info: list
            Different rings that contain the pair of atoms
        """
        features = []
        path_bonds = []
        path_length = len(path_atoms)
        for path_idx in range(path_length - 1):
            bond = mol.GetBondBetweenAtoms(path_atoms[path_idx],
                                           path_atoms[path_idx + 1])
            if bond is None:
                import warnings
                warnings.warn('Valid idx of bonds must be passed')
            path_bonds.append(bond)

        for path_idx in range(self.max_length):
            if path_idx < len(path_bonds):
                bond_type = get_bond_type_one_hot(path_bonds[path_idx])
                conjugacy = get_bond_is_conjugated_one_hot(path_bonds[path_idx])
                ring_attach = get_bond_is_in_same_ring_one_hot(path_bonds[path_idx])
                features.append(np.concatenate([bond_type, conjugacy, ring_attach]))
            else:
                features.append(np.zeros(6))

        if path_length + 1 > self.max_length:
            path_length = self.max_length + 1
        position_feature = np.zeros(self.max_length + 2)
        position_feature[path_length] = 1
        features.append(position_feature)
        if ring_info:
            rfeat = [
                one_hot_encode(r, allowable_set=self.RING_TYPES) for r in ring_info
            ]
            # The 1.0 float value represents True Boolean
            rfeat = [1.0] + np.any(rfeat, axis=0).tolist()
            features.append(rfeat)
        else:
            # This will return a boolean vector with all entries False
            features.append([0.0] +
                            one_hot_encode(ring_info, allowable_set=self.RING_TYPES))
        return np.concatenate(features, axis=0)

    def _pagtn_edge_featurizer(self,
                               mol: RDKitMol) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate bond features from RDKit mol object.
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit mol object.
        Returns
        -------
        np.ndarray
          Source and Destination node indexes of each bond.
        np.ndarray
          numpy vector of bond features.
        """
        n_atoms = mol.GetNumAtoms()
        # To get the shortest paths between two nodes.
        paths_dict = compute_all_pairs_shortest_path(mol)
        # To get info if two nodes belong to the same ring.
        rings_dict = compute_pairwise_ring_info(mol)
        # Featurizer
        feats = []
        src = []
        dest = []
        for i in range(n_atoms):
            for j in range(n_atoms):
                src.append(i)
                dest.append(j)

                if (i, j) not in paths_dict:
                    feats.append(np.zeros(7 * self.max_length + 7))
                    continue
                ring_info = rings_dict.get(self.ordered_pair(i, j), [])
                feats.append(self._edge_features(mol, paths_dict[(i, j)], ring_info))

        return np.array([src, dest], dtype=int), np.array(feats, dtype=float)

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        """Calculate molecule graph features from RDKit mol object.
    Parameters
    ----------
    datapoint: rdkit.Chem.rdchem.Mol
      RDKit mol object.
    Returns
    -------
    graph: GraphData
      A molecule graph with some features.
    """
        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )

        node_features = np.asarray(
            [self._pagtn_atom_featurizer(atom) for atom in datapoint.GetAtoms()],
            dtype=float)
        edge_index, edge_features = self._pagtn_edge_featurizer(datapoint)
        graph = GraphData(node_features, edge_index, edge_features)
        return graph

    def _featurize_dataframe(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        """Calculate molecule graph features from RDKit mol object.
    Parameters
    ----------
    datapoint: rdkit.Chem.rdchem.Mol
      RDKit mol object.
    Returns
    -------
    graph: GraphData
      A molecule graph with some features.
    """
        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )

        node_features = np.asarray(
            [self._pagtn_atom_featurizer(atom) for atom in datapoint.GetAtoms()],
            dtype=float)
        edge_index, edge_features = self._pagtn_edge_featurizer(datapoint)
        graph = GraphData(node_features, edge_index, edge_features)
        return graph



class TorchMolGraphConvFeaturizer(MolecularFeaturizer):
    """This class is a featurizer of pytorch-geometric graph convolution networks for molecules.
      The default node(atom) and edge(bond) representations are based on
      `WeaveNet paper <https://arxiv.org/abs/1603.00856>`_. If you want to use your own representations,
      you could use this class as a guide to define your original Featurizer. In many cases, it's enough
      to modify return values of `construct_atom_feature` or `construct_bond_feature`.
      The default node representation are constructed by concatenating the following values,
      and the feature length is 30.
      - Atom type: A one-hot vector of this atom, "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other atoms".
      - Formal charge: Integer electronic charge.
      - Hybridization: A one-hot vector of "sp", "sp2", "sp3".
      - Hydrogen bonding: A one-hot vector of whether this atom is a hydrogen bond donor or acceptor.
      - Aromatic: A one-hot vector of whether the atom belongs to an aromatic ring.
      - Degree: A one-hot vector of the degree (0-5) of this atom.
      - Number of Hydrogens: A one-hot vector of the number of hydrogens (0-4) that this atom connected.
      - Chirality: A one-hot vector of the chirality, "R" or "S". (Optional)
      - Partial charge: Calculated partial charge. (Optional)
      The default edge representation are constructed by concatenating the following values,
      and the feature length is 11.
      - Bond type: A one-hot vector of the bond type, "single", "double", "triple", or "aromatic".
      - Same ring: A one-hot vector of whether the atoms in the pair are in the same ring.
      - Conjugated: A one-hot vector of whether this bond is conjugated or not.
      - Stereo: A one-hot vector of the stereo configuration of a bond.
      If you want to know more details about features, please check the paper [1]_ and
      utilities in deepchem.utils.molecule_feature_utils.py.
      Examples
      --------
      >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
      >>> featurizer = TorchMolGraphConvFeaturizer(use_edges=True)
      >>> out = featurizer.featurize(smiles)
      >>> type(out[0])
      <class 'jaqpotpy.descriptors.graph.GraphData'>
      >>> out[0].num_node_features
      30
      >>> out[0].num_edge_features
      11
      References
      ----------
      .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints."
         Journal of computer-aided molecular design 30.8 (2016):595-608.
      Note
      ----
      This class requires RDKit to be installed.
      """
    import torch
    from torch_geometric.data import Data

    @property
    def __name__(self):
        return 'TorchMolGraphConvFeaturizer'

    def __init__(self,
                 use_edges: bool = False,
                 use_chirality: bool = False,
                 use_partial_charge: bool = False):
        """
        Parameters
        ----------
        use_edges: bool, default False
          Whether to use edge features or not.
        use_chirality: bool, default False
          Whether to use chirality information or not.
          If True, featurization becomes slow.
        use_partial_charge: bool, default False
          Whether to use partial charge data or not.
          If True, this featurizer computes gasteiger charges.
          Therefore, there is a possibility to fail to featurize for some molecules
          and featurization becomes slow.
        """
        self.use_edges = use_edges
        self.use_partial_charge = use_partial_charge
        self.use_chirality = use_chirality

    def __getitem__(self):
        return self

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> Data:
        """Calculate molecule graph features from RDKit mol object.
        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
          RDKit mol object.
        Returns
        -------
        graph: GraphData
          A molecule graph with some features.
        """
        assert datapoint.GetNumAtoms(
        ) > 1, "More than one atom should be present in the molecule for this featurizer to work."
        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )

        if self.use_partial_charge:
            try:
                datapoint.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
            except:
                # If partial charges were not computed
                try:
                    from rdkit.Chem import AllChem
                    AllChem.ComputeGasteigerCharges(datapoint)
                except ModuleNotFoundError:
                    raise ImportError("This class requires RDKit to be installed.")

        # construct atom (node) feature
        h_bond_infos = construct_hydrogen_bonding_info(datapoint)
        atom_features = np.asarray(
            [
                _construct_atom_feature(atom, h_bond_infos, self.use_chirality,
                                        self.use_partial_charge)
                for atom in datapoint.GetAtoms()
            ],
            dtype=float,
        )

        # construct edge (bond) index
        src, dest = [], []
        for bond in datapoint.GetBonds():
            # add edge list considering a directed graph
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]

        # construct edge (bond) feature
        bond_features = None  # deafult None
        if self.use_edges:
            features = []
            for bond in datapoint.GetBonds():
                features += 2 * [_construct_bond_feature(bond)]
            bond_features = np.asarray(features, dtype=float)
        import torch
        from torch_geometric.data import Data
        return Data(
            node_features=torch.FloatTensor(atom_features),
            edge_index=torch.LongTensor(np.asarray([src, dest], dtype=int)),
            edge_features=bond_features)

    def _get_column_names(self, **kwargs) -> list:
        """
        Return the column names
        """
        names = ['TorchMoleculeGraph']
        return names

    def _featurize_dataframe(self, datapoint: RDKitMol, **kwargs) -> Data:
        return self._featurize(datapoint, **kwargs)


class AttentiveFPFeaturizer(MolecularFeaturizer):
    """
    This class is a featurizer that is introduced in the publication of Xiong et. al. [1]
    for Attentive FP GNNs. It uses a specific a specific fingerprint for the nodes and a different one
    for the edges. In more details, the node features are:
    - One hot encoding of the atom type. The supported atom types include:
        "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "Te", "I", "At", and "other"
    - One hot encoding of the atom degree. The supported possibilities include "0 - 5"
    - The formal charge of the atom in the node
    - The number of the radical electrons of the atom
    - One hot ecoding of the atom's hybridization. Supported hybridizations are "SP", "SP2",
      "SP3", "SP3D", "SP3D2", and "other"
    - Indication on whether the atom is aromatic or not
    - One hot encoding of the number f total hydrogens on the atom. Supported possibilities are 0 - 4.
    - Indication on whether the atom is chiral center or not
    - One hot encoding of the atom chirality type. The supported possibilities include "R", and "S".

    The edge features include:
    - One hote encoding of the bond type. The supported types are "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"
    - Indication on whether the bond is conjugated or not
    - Indication on whether the bond is a ring or not
    - One hote encoding of the bond stereo configuration. The supported types are "STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"

    Examples
    --------
    >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
    >>> featurizer = AttentiveFPFeaturizer(use_edges=True)
    >>> out = featurizer.featurize(smiles)
    >>> type(out[0])
    <class 'jaqpotpy.descriptors.graph.GraphData'>
    >>> out[0].num_node_features
    30
    >>> out[0].num_edge_features
    11
    References
    ----------
    .. [1] Xiong Z, Wang D, Liu X, Zhong F, Wan X, Li X, Li Z, Luo X, Chen K, Jiang H, Zheng M. Pushing the Boundaries
           of Molecular Representation for Drug Discovery with the Graph Attention Mechanism. J Med Chem. 2020 Aug 27;63(16):8749-8760.
           doi: 10.1021/acs.jmedchem.9b00959. Epub 2019 Aug 27. PMID: 31408336.
    Note
    ----
    This class requires RDKit to be installed.
    """
    from torch_geometric.data import Data

    @property
    def __name__(self):
        return 'AttentiveFPFeaturizer'

    def __init__(self, use_loops: bool = False):
        """
        Parameters
        ----------
        use_loops: bool, (default False)
          Whether self loops will be added.
        """
        self.use_loops = use_loops

    def __getitem__(self):
        return self

    def _featurize_node(self, atom):
        from rdkit import Chem
        feature_funcs = [
            partial(get_atom_type_one_hot, allowable_set=[
                'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S',
                'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'], include_unknown_set=True),
            partial(get_atom_degree_one_hot, allowable_set=list(range(6)), include_unknown_set=False),
            get_atom_formal_charge,
            get_atom_num_radical_electrons,
            partial(get_atom_hybridization_one_hot,
                    allowable_set = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'],
                    include_unknown_set=True),
            get_atom_is_aromatic,
            partial(get_atom_total_num_Hs_one_hot, include_unknown_set=False),
            get_atom_is_chiral_center,
            get_atom_chirality_one_hot
        ]

        features = []
        for func in feature_funcs:
            features.extend(func(atom))

        return np.array(features)


    def _featurize_edge(self, bond, mol):
        from rdkit import Chem
        feature_funcs = [
            get_bond_type_one_hot,
            get_bond_is_conjugated_one_hot,
            get_bond_is_in_same_ring_one_hot,
            partial(get_bond_stereo_one_hot, allowable_set=['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE'],
                    include_unknown_set = False)
        ]

        features = []
        for func in feature_funcs:
            features.extend(func(bond))

        features = np.array(features)
        features = np.stack((features, features.copy()))

        if self.use_loops and mol.GetNumBonds() > 0:
            import torch
            num_atoms = mol.GetNumAtoms()
            feats = torch.cat([features, torch.zeros(features.shape[0], 1)], dim=1)
            self_loop_feats = torch.zeros(num_atoms, feats.shape[1])
            self_loop_feats[:, -1] = 1
            feats = torch.cat([feats, self_loop_feats], dim=0)
            features = feats

        if self.use_loops and mol.GetNumBonds() == 0:
            import torch
            num_atoms = mol.GetNumAtoms()
            dummy = Chem.MolFromSmiles('CO')
            features = self._featurize_edge(dummy.GetBonds()[0], dummy)
            feats = torch.zeros(num_atoms, features.shape[1])
            feats[:, -1] = 1
            features = feats

        return features

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        """Calculate molecule graph features from RDKit mol object.
        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
          RDKit mol object.
        Returns
        -------
        graph: GraphData
          A molecule graph with some features.
        """
        assert datapoint.GetNumAtoms(
        ) > 1, "More than one atom should be present in the molecule for this featurizer to work."
        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )


        # construct atom (node) feature
        node_features = np.asarray([self._featurize_node(atom) for atom in datapoint.GetAtoms()], dtype=float)

        # construct edge (bond) index
        src, dest = [], []
        for bond in datapoint.GetBonds():
            # add edge list considering a directed graph
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]

        # construct edge (bond) feature
        bond_features = np.asarray([self._featurize_edge(bond, datapoint) for bond in datapoint.GetBonds()], dtype=float)  # deafult None
        bond_features = bond_features.reshape(bond_features.shape[0] * bond_features.shape[1], bond_features.shape[2])
        # bond_features = torch.LongTensor(bond_features)

        import torch
        return GraphData(
            node_features= node_features,
            edge_index= np.asarray([src, dest], dtype=int),
            edge_features= bond_features)

    def _get_column_names(self, **kwargs) -> list:
        """
        Return the column names
        """
        names = ['AttentiveFP']
        return names

    def _featurize_dataframe(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        return self._featurize(datapoint, **kwargs)


