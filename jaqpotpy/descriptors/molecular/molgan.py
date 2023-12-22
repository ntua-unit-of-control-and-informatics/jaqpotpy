import logging
import numpy as np
from jaqpotpy.utils.types import RDKitBond, RDKitMol, List, OneOrMany
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from jaqpotpy.cfg import config
from typing import Optional

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger(__name__)


class GraphMatrix:
  """
  This is class used to store data for MolGAN neural networks.
  Parameters
  ----------
  node_features: np.ndarray
    Node feature matrix with shape [num_nodes, num_node_features]
  edge_features: np.ndarray,
    Edge feature matrix with shape [num_nodes, num_nodes]
  Returns
  -------
  graph: GraphMatrix
    A molecule graph with some features.
  """

  def __init__(self, adjacency_matrix: np.ndarray, node_features: np.ndarray):
    self.adjacency_matrix = adjacency_matrix
    self.node_features = node_features

  @property
  def __name__(self):
    return 'GraphMatrix'

  def __getitem__(self):
    return self

  # @property
  # def adjacency_matrix(self):
  #   return self.adjacency_matrix
  #
  # @property
  # def node_features(self):
  #   return self.node_features


class MolGanFeaturizer(MolecularFeaturizer):
  """
  Featurizer for MolGAN de-novo molecular generation [1]_.
  The default representation is in form of GraphMatrix object.
  It is wrapper for two matrices containing atom and bond type information.
  The class also provides reverse capabilities.
  Examples
  --------
  >>> import jaqpotpy as jp
  >>> from rdkit import Chem
  >>> rdkit_mol, smiles_mol = Chem.MolFromSmiles('CCC'), 'C1=CC=CC=C1'
  >>> molecules = [rdkit_mol, smiles_mol]
  >>> featurizer = jp.descriptors.molecular.MolGanFeaturizer()
  >>> features = featurizer.featurize(molecules)
  >>> len(features) # 2 molecules
  2
  >>> type(features[0])
  <class 'deepchem.feat.molecule_featurizers.molgan_featurizer.GraphMatrix'>
  >>> molecules = featurizer.defeaturize(features) # defeaturization
  >>> type(molecules[0])
  <class 'rdkit.Chem.rdchem.Mol'>
  """

  @property
  def __name__(self):
    return 'MolGan'

  def __getitem__(self):
    return self

  def __init__(
      self,
      max_atom_count: int = 9,
      kekulize: bool = True,
      sanitize: bool = False,
      bond_labels: List[RDKitBond] = None,
      atom_labels: List[int] = None,
  ):
    """
    Parameters
    ----------
    max_atom_count: int, default 9
      Maximum number of atoms used for creation of adjacency matrix.
      Molecules cannot have more atoms than this number
      Implicit hydrogens do not count.
    kekulize: bool, default True
      Should molecules be kekulized.
      Solves number of issues with defeaturization when used.
    bond_labels: List[RDKitBond]
      List of types of bond used for generation of adjacency matrix
    atom_labels: List[int]
      List of atomic numbers used for generation of node features
    References
    ---------
    .. [1] Nicola De Cao et al. "MolGAN: An implicit generative model for
       small molecular graphs" (2018), https://arxiv.org/abs/1805.11973
    """

    self.max_atom_count = max_atom_count

    self.BOND_DIM = 5
    self.kekulize = kekulize
    self.sanitize = sanitize

    try:
      from rdkit import Chem
    except ModuleNotFoundError:
      raise ImportError("This class requires RDKit to be installed.")

    self.bond_mapping = {
      "SINGLE": 0,
      0: Chem.BondType.SINGLE,
      "DOUBLE": 1,
      1: Chem.BondType.DOUBLE,
      "TRIPLE": 2,
      2: Chem.BondType.TRIPLE,
      "AROMATIC": 3,
      3: Chem.BondType.AROMATIC,
    }

    # bond labels
    if bond_labels is None:
      self.bond_labels = [
          Chem.rdchem.BondType.ZERO,
          Chem.rdchem.BondType.SINGLE,
          Chem.rdchem.BondType.DOUBLE,
          Chem.rdchem.BondType.TRIPLE,
          Chem.rdchem.BondType.AROMATIC,
      ]
    else:
      self.bond_labels = bond_labels

    self.SMILE_CHARSET = ["C", "N", "O", "F", "B", "I", "H", "S", "P", "Cl", "Br"]

    # atom labels
    if atom_labels is None:
      self.atom_labels = [0, 6, 7, 8, 9, 5, 53, 1, 16, 15, 17, 35]  # C,N,O,F,B, I, H, S, P, Cl, Br
    else:
      self.atom_labels = atom_labels

    # create bond encoders and decoders
    self.bond_encoder = {l: i for i, l in enumerate(self.bond_labels)}
    self.bond_decoder = {i: l for i, l in enumerate(self.bond_labels)}
    # create atom encoders and decoders
    self.atom_encoder = {l: i for i, l in enumerate(self.atom_labels)}
    self.atom_decoder = {i: l for i, l in enumerate(self.atom_labels)}

    self.MAX_ATOMS = len(self.atom_labels)

  def _featurize(self, datapoint: RDKitMol, **kwargs) -> Optional[GraphMatrix]:
    """
    Calculate adjacency matrix and nodes features for RDKitMol.
    It strips any chirality and charges
    Parameters
    ----------
    datapoint: rdkit.Chem.rdchem.Mol
      RDKit mol object.
    Returns
    -------
    graph: GraphMatrix
      A molecule graph with some features.
    """

    try:
      from rdkit import Chem
    except ModuleNotFoundError:
      raise ImportError("This method requires RDKit to be installed.")
    if 'mol' in kwargs:
      datapoint = kwargs.get("mol")
      raise DeprecationWarning(
          'Mol is being phased out as a parameter, please pass "datapoint" instead.'
      )

    if self.kekulize:
      Chem.Kekulize(datapoint)

    adjacency = np.zeros((self.BOND_DIM, self.max_atom_count, self.max_atom_count), "float32")
    features = np.zeros((self.max_atom_count, self.MAX_ATOMS), "float32")

    # loop over each atom in molecule
    for atom in datapoint.GetAtoms():
        i = atom.GetIdx()
        atom_type = self.atom_encoder[atom.GetAtomicNum()]
        features[i] = np.eye(self.MAX_ATOMS)[atom_type]
        # loop over one-hop neighbors
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            bond = datapoint.GetBondBetweenAtoms(i, j)
            bond_type_idx = self.bond_mapping[bond.GetBondType().name]
            adjacency[bond_type_idx, [i, j], [j, i]] = 1

    # Where no bond, add 1 to last channel (indicating "non-bond")
    # Notice: channels-first
    adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1

    # Where no atom, add 1 to last column (indicating "non-atom")
    features[np.where(np.sum(features, axis=1) == 0)[0], -1] = 1

    degree = np.sum(
        adjacency[:datapoint.GetNumAtoms(), :datapoint.GetNumAtoms()], axis=-1)


    graph = GraphMatrix(adjacency, features)
    # features[np.where(np.sum(X, axis=1) == 0)[0], -1] = 1
    return graph
    # return graph if (degree > 0).all() else None


    # A = np.zeros(
    #     shape=(self.max_atom_count, self.max_atom_count), dtype=np.float32)
    # bonds = datapoint.GetBonds()
    # begin, end = [b.GetBeginAtomIdx() for b in bonds], [
    #     b.GetEndAtomIdx() for b in bonds
    # ]
    # bond_type = [self.bond_encoder[b.GetBondType()] for b in bonds]
    # A[begin, end] = bond_type
    # A[end, begin] = bond_type
    # adjacency[bond_type, [begin, end], [end, begin]] = 1
    # adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1
    # degree = np.sum(
    #     A[:datapoint.GetNumAtoms(), :datapoint.GetNumAtoms()], axis=-1)
    # X = np.array(
    #     [
    #         self.atom_encoder[atom.GetAtomicNum()]
    #         for atom in datapoint.GetAtoms()
    #     ] + [0] * (self.max_atom_count - datapoint.GetNumAtoms()),
    #     dtype=np.int32,
    # )
    #
    # x_d = np.eye(len(self.atom_labels))[X]
    # # features[np.where(np.sum(x_d, axis=1) == 0)[0], -1] = 1
    # # features[i] = np.eye(ATOM_DIM)[atom_type]
    # # graph = GraphMatrix(A, X)
    # graph = GraphMatrix(adjacency, x_d)
    # # features[np.where(np.sum(X, axis=1) == 0)[0], -1] = 1
    # # return graph
    # return graph if (degree > 0).all() else None

  def _defeaturize(self,
                   graph_matrix: GraphMatrix,
                   sanitize: bool = True,
                   cleanup: bool = True) -> RDKitMol:
    """
    Recreate RDKitMol from GraphMatrix object.
    Same featurizer need to be used for featurization and defeaturization.
    It only recreates bond and atom types, any kind of additional features
    like chirality or charge are not included.
    Therefore, any checks of type: original_smiles == defeaturized_smiles
    will fail on chiral or charged compounds.
    Parameters
    ----------
    graph_matrix: GraphMatrix
      GraphMatrix object.
    sanitize: bool, default True
      Should RDKit sanitization be included in the process.
    cleanup: bool, default True
      Splits salts and removes compounds with "*" atom types
    Returns
    -------
    mol: RDKitMol object
      RDKitMol object representing molecule.
    """

    try:
      from rdkit import Chem
      import torch
    except ModuleNotFoundError:
      raise ImportError("This method requires RDKit to be installed.")

    if not isinstance(graph_matrix, GraphMatrix):
      return None

    node_labels = graph_matrix.node_features
    edge_labels = graph_matrix.adjacency_matrix

    # edge_labels, node_labels = torch.max(torch.from_numpy(edge_labels), -1)[1].cpu().numpy()\
    #   , torch.max(torch.from_numpy(node_labels), -1)[1].cpu().numpy()

    mol = Chem.RWMol()

    # Remove "no atoms" & atoms with no bonds
    keep_idx = np.where(
        (np.argmax(node_labels, axis=1) != len(self.atom_labels) - 1)
        & (np.sum(edge_labels[:-1], axis=(0, 1)) != 0)
    )[0]
    features = node_labels[keep_idx]
    adjacency = edge_labels[:, keep_idx, :][:, :, keep_idx]

    # Add atoms to molecule
    for atom_type_idx in np.argmax(features, axis=1):
        atomic_number = self.atom_decoder[atom_type_idx]
        atom = Chem.Atom(atomic_number)
        mol.AddAtom(atom)

    # Add bonds between atoms in molecule; based on the upper triangles
    # of the [symmetric] adjacency tensor
    # (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
    # for (bond_ij, atom_i, atom_j) in zip(bonds_ij, atoms_i, atoms_j):
    #     if atom_i == atom_j or bond_ij == len(self.bond_labels) - 1:
    #         continue
    #     bond_type = self.bond_decoder[bond_ij]
    #     try:
    #       mol.AddBond(int(atom_i), int(atom_j), bond_type)
    #     except Exception as e:
    #       continue

    (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
    for (bond_ij, atom_i, atom_j) in zip(bonds_ij, atoms_i, atoms_j):
        if atom_i == atom_j or bond_ij == self.BOND_DIM - 1:
            continue
        bond_type = self.bond_mapping[bond_ij]
        mol.AddBond(int(atom_i), int(atom_j), bond_type)


    # mol = Chem.RWMol()
    #
    # for node_label in node_labels:
    #   mol.AddAtom(Chem.Atom(self.atom_decoder[node_label]))
    #
    # for start, end in zip(*np.nonzero(edge_labels)):
    #   if start > end:
    #     mol.AddBond(int(start), int(end), self.bond_decoder[edge_labels[start, end]])

    if sanitize:
      try:
        Chem.SanitizeMol(mol)
      except Exception:
        mol = None


    if cleanup:
      try:
        smiles = Chem.MolToSmiles(mol)
        smiles = max(smiles.split("."), key=len)
        if "*" not in smiles:
          mol = Chem.MolFromSmiles(smiles)
        else:
          mol = None
      except Exception:
        mol = None

    return mol

  def defeaturize(self, graphs: OneOrMany[GraphMatrix],
                  log_every_n: int = 1000) -> np.ndarray:
    """
    Calculates molecules from corresponding GraphMatrix objects.
    Parameters
    ----------
    graphs: GraphMatrix / iterable
      GraphMatrix object or corresponding iterable
    log_every_n: int, default 1000
      Logging messages reported every `log_every_n` samples.
    Returns
    -------
    features: np.ndarray
      A numpy array containing RDKitMol objext.
    """

    # Special case handling of single molecule
    if isinstance(graphs, GraphMatrix):
      graphs = [graphs]
    else:
      # Convert iterables to list
      graphs = list(graphs)

    molecules = []
    for i, gr in enumerate(graphs):
      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)
      try:
        molecules.append(self._defeaturize(gr))
      except Exception as e:
        if config.verbose is True:
          logger.warning(
              "Failed to defeaturize datapoint %d, %s. Appending empty array",
              i,
              gr,
          )
          logger.warning("Exception message: {}".format(e))
        else:
          continue
        molecules.append(np.array([]))

    return np.asarray(molecules)

  def _get_column_names(self):
    return ["MolGanGraphs"]

  def _featurize_dataframe(self, datapoint: RDKitMol, **kwargs):
    return self._featurize(datapoint,  **kwargs)
