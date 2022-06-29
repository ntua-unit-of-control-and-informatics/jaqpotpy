from jaqpotpy.descriptors.base_classes import MaterialFeaturizer
from jaqpotpy.descriptors.graph.graph_data import GraphData
from pymatgen.core.structure import Structure
import numpy as np
import json
from typing import Union, Tuple
import pandas as pd


class CrystalGraphCNN(MaterialFeaturizer):
  """
  Calculate structure graph features for crystals.
  Based on the implementation in Crystal Graph Convolutional
  Neural Networks (CGCNN). The method constructs a crystal graph
  representation including atom features and bond features (neighbor
  distances). Neighbors are determined by searching in a sphere around
  atoms in the unit cell. A Gaussian filter is applied to neighbor distances.
  All units are in angstrom.
  This featurizer requires the dependency pymatgen. It may
  be useful when 3D coordinates are available and when using graph
  network models and crystal graph convolutional networks.
  See [1]_ for more details.
  References
  ----------
  .. [1] T. Xie and J. C. Grossman, "Crystal graph convolutional
     neural networks for an accurate and interpretable prediction
     of material properties", Phys. Rev. Lett.
  Examples
  --------
  >>> import jaqpotpy as jt
  >>> from pymatgen.core.lattice import Lattice
  >>> from pymatgen.core.structure import Structure
  >>> lattice = Lattice.cubic(4.2)
  >>> struct = Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
  >>> featurizer = jt.descriptors.material.CrystalGraphCNN()
  >>> features = featurizer.featurize([struct])
  >>> type(features[0])
  <class 'jaqpotpy.descriptors.material.GraphData'>

  Note
  ----
  This class requires matminer and Pymatgen to be installed.
  `NaN` feature values are automatically converted to 0 by this featurizer.
  """

  def __init__(self, radius: float = 8.0, max_neighbors: int = 12, step:float = 0.2):
    """
     Parameters
    ----------
    radius: float (default 8.0)
      Radius of sphere for finding neighbors of atoms in unit cell.
    max_neighbors: int (default 12)
      Maximum number of neighbors to consider when constructing graph.
    step: float (default 0.2)
      Step size for Gaussian filter. This value is used when building edge features.
    """
    self.radius = radius
    self.max_neighbors = int(max_neighbors)
    self.step = step
    from jaqpotpy.descriptors.material.datafiles.atom_init import atom_init_json

    # print(atom_init_json)
    # with open('jaqpotpy/descriptors/material/datafiles/atom_init.json', 'r') as f:
    #     atom_init_json = json.load(f)

    self.atom_features = {
        int(key): np.array(value, dtype=np.float32)
        for key, value in atom_init_json.items()
    }
    self.valid_atom_number = set(self.atom_features.keys())

  @property
  def __name__(self):
    return 'CrystalGraphCNN'

  def __getitem__(self):
    return self


  def _featurize(self, datapoint: Union[str, Structure], **kwargs) -> GraphData:
    """
    Calculate crystal graph features from pymatgen structure.

    Parameters
    ----------
    datapoint: str or pymatgen.core.structure object
      Either the path of a file of a material (i.e. extxyz, cif etc.)
      from wich the structure will be created, or the Structure itself.
    Returns
    -------
    feats: np.ndarray
      Vector of properties and statistics derived from chemical
      stoichiometry. Some values may be NaN.
    """

    if isinstance(datapoint, str):
        if datapoint.split('.')[-1] == 'extxyz':
            from ase.io import read
            xyz = read(datapoint)
            lat = np.array(xyz.get_cell())
            sym = xyz.get_chemical_symbols()
            pos = xyz.arrays['positions']
            struct = Structure(lat, sym, pos)
        else:
            struct = Structure.from_file(datapoint)
    else:
        struct = datapoint

    node_features = self._get_node_features(struct)
    edge_index, edge_features = self._get_edge_features_and_index(struct)
    graph = GraphData(node_features, edge_index, edge_features)
    return graph


  def _featurize_dataframe(self, datapoint: Union[str, Structure], **kwargs) -> pd.DataFrame:
    """
    Calculate crystal graph features from pymatgen structure.

    Parameters
    ----------
    datapoint: str or pymatgen.core.structure object
      Either the path of a file of a material (i.e. extxyz, cif etc.)
      from wich the structure will be created, or the Structure itself.
    Returns
    -------
    feats: np.ndarray
      Vector of properties and statistics derived from chemical
      stoichiometry. Some values may be NaN.
    """

    df = pd.DataFrame(index=[0], columns=['MaterialGraph'])
    df.at[0, 'MaterialGraph'] = self._featurize(datapoint)
    return df

  def _get_node_features(self, struct: Structure) -> np.ndarray:
      """
      Get the node feature from `atom_init.json`. The `atom_init.json` was collected
      from `data/sample-regression/atom_init.json` in the CGCNN repository.
      Parameters
      ----------
      struct: pymatgen.core.Structure
        A periodic crystal composed of a lattice and a sequence of atomic
        sites with 3D coordinates and elements.
      Returns
      -------
      node_features: np.ndarray
        A numpy array of shape `(num_nodes, 92)`.
      """
      node_features = []
      for site in struct:
          # check whether the atom feature exists or not
          assert site.specie.number in self.valid_atom_number
          node_features.append(self.atom_features[site.specie.number])
      return np.vstack(node_features).astype(float)

  def _get_edge_features_and_index(self, struct: Structure) -> Tuple[np.ndarray, np.ndarray]:
      """
      Calculate the edge feature and edge index from pymatgen structure.
      Parameters
      ----------
      struct: pymatgen.core.Structure
        A periodic crystal composed of a lattice and a sequence of atomic
        sites with 3D coordinates and elements.
      Returns
      -------
      edge_idx np.ndarray, dtype int
        A numpy array of shape with `(2, num_edges)`.
      edge_features: np.ndarray
        A numpy array of shape with `(num_edges, filter_length)`. The `filter_length` is
        (self.radius / self.step) + 1. The edge features were built by applying gaussian
        filter to the distance between nodes.
      """

      neighbors = struct.get_all_neighbors(self.radius, include_index=True)
      neighbors = [sorted(n, key=lambda x: x[1]) for n in neighbors]

      # construct bi-directed graph
      src_idx, dest_idx = [], []
      edge_distances = []
      for node_idx, neighbor in enumerate(neighbors):
          neighbor = neighbor[:self.max_neighbors]
          src_idx.extend([node_idx] * len(neighbor))
          dest_idx.extend([site[2] for site in neighbor])
          edge_distances.extend([site[1] for site in neighbor])

      edge_idx = np.array([src_idx, dest_idx], dtype=int)
      edge_features = self._gaussian_filter(np.array(edge_distances, dtype=float))
      return edge_idx, edge_features

  def _gaussian_filter(self, distances: np.ndarray) -> np.ndarray:
      """
      Apply Gaussian filter to an array of interatomic distances.
      Parameters
      ----------
      distances : np.ndarray
        A numpy array of the shape `(num_edges, )`.
      Returns
      -------
      expanded_distances: np.ndarray
        Expanded distance tensor after Gaussian filtering.
        The shape is `(num_edges, filter_length)`. The `filter_length` is
        (self.radius / self.step) + 1.
      """

      filt = np.arange(0, self.radius + self.step, self.step)

      # Increase dimension of distance tensor and apply filter
      expanded_distances = np.exp(-(distances[..., np.newaxis] - filt) ** 2 / self.step ** 2)

      return expanded_distances

