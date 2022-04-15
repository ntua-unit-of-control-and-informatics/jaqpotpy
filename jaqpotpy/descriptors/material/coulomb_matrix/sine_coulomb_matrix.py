from jaqpotpy.descriptors.base_classes import MaterialFeaturizer
from pymatgen.core.structure import Structure, PeriodicSite
from typing import Union, Tuple
import numpy as np
import scipy.constants as const
import pandas as pd

ANG_TO_BOHR = const.value("Angstrom star") / const.value("Bohr radius")


class SineCoulombMatrix(MaterialFeaturizer):
  """
  Create a fixed vector of length 118, containing rwa, fractional
  element compostitions in a compound.

  References
  ----------
  Jha, D., Ward, L., Paul, A. et al. Sci Rep 8, 17593 (2018).
     https://doi.org/10.1038/s41598-018-35934-y

  Examples
  --------
  >>> import jaqpotpy as jt
  >>> from pymatgen.core.lattice import Lattice
  >>> from pymatgen.core.structure import Structure
  >>> lattice = Lattice.cubic(4.2)
  >>> struct = Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
  >>> featurizer = jt.descriptors.material.SineCoulombMatrix()
  >>> features = featurizer.featurize(struct)
  >>> type(features[0])
  <class 'numpy.ndarray'>
  >>> features[0].shape
  (118,)
  Note
  ----
  This class requires matminer and Pymatgen to be installed.
  `NaN` feature values are automatically converted to 0 by this featurizer.
  """

  def __init__(self, max_atoms: int = 100, diag_elements: bool = True, flatten: bool = False):
    """
    Create a Sine Coulomb Matrix for a pymatgen.core.Structure

    Parameters
    ----------
    max_atoms: int (default 100)
      Maximum number of atoms for any crystal in the dataset. Used to
      pad the Coulomb matrix.
    diag_elems (bool): flag indication whether (True, default) to use
      the original definition of the diagonal elements; if set to False,
      the diagonal elements are set to 0
    flatten: bool (default True)
      Return flattened vector of matrix eigenvalues.

      References
      ----------
      Faber et al. "Crystal Structure Representations for Machine
         Learning Models of Formation Energies", Inter. J. Quantum Chem.
         115, 16, 2015. https://arxiv.org/abs/1503.07406
    """
    self.diag_elements = diag_elements
    self.flatten = flatten
    self.max_atoms = max_atoms
    return

  @property
  def __name__(self):
    return 'SineCoulombMatrix'

  def __getitem__(self):
    return self


  def _featurize(self, datapoint: Union[str, Structure], **kwargs) -> np.ndarray:
    """
    Calculate sine coulomb matrix for a structure.

    Parameters
    ----------
    datapoint: str (path of a file) or pymatgen.core.Structure
      Structure or path of a file to parse into a Structure

    Returns
    -------
    feats: np.ndarray
      Vector containing the sine coulomb matrix of a structure.
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

    sites = struct.sites
    atomic_numbers = np.array([site.specie.Z for site in sites])
    sin_mat = np.zeros((len(sites), len(sites)))
    coords = np.array([site.frac_coords for site in sites])
    lattice = struct.lattice.matrix

    for i in range(len(sin_mat)):
        for j in range(len(sin_mat)):
            if i == j:
                if self.diag_elements:
                    sin_mat[i][i] = 0.5 * atomic_numbers[i] ** 2.4
            elif i < j:
                vec = coords[i] - coords[j]
                coord_vec = np.sin(np.pi * vec) ** 2
                trig_dist = np.linalg.norm((np.asmatrix(coord_vec) * lattice).A1) * ANG_TO_BOHR
                sin_mat[i][j] = atomic_numbers[i] * atomic_numbers[j] / trig_dist
            else:
                sin_mat[i][j] = sin_mat[j][i]

    if self.flatten:
        eigs, _ = np.linalg.eig([sin_mat])
        zeros = np.zeros(self.max_atoms)
        zeros[:len(eigs[0])] = eigs[0]
        features = zeros
    else:
        features = pad_array([sin_mat], self.max_atoms)

    features = np.asarray(features)
    return features



  def _featurize_dataframe(self, datapoint: Union[str, Structure], **kwargs) -> pd.DataFrame:
    """
    Calculate sine coulomb matrix for a structure.

    Parameters
    ----------
    datapoint: str (path of a file) or pymatgen.core.Structure
      Structure or path of a file to parse into a Structure

    Returns
    -------
    feats: pd.DataFrame
      Vector containing the sine coulomb matrix of a structure.
    """
    self.flatten = True
    data = self._featurize(datapoint)
    columns = ['Feat_{}'.format(str(i)) for i in range(self.max_atoms)]
    return pd.DataFrame([data], columns=columns, index=[0])

def pad_array(x: list,
              shape: Union[Tuple, int],
              fill: float = 0.0,
              both: bool = False) -> np.ndarray:
  """
  Pad an array with a fill value.
  Parameters
  ----------
  x: np.ndarray
    A numpy array.
  shape: Tuple or int
    Desired shape. If int, all dimensions are padded to that size.
  fill: float, optional (default 0.0)
    The padded value.
  both: bool, optional (default False)
    If True, split the padding on both sides of each axis. If False,
    padding is applied to the end of each axis.
  Returns
  -------
  np.ndarray
    A padded numpy array
  """
  x = np.asarray(x)
  if not isinstance(shape, tuple):
    shape = tuple(shape for _ in range(x.ndim))
  pad = []
  for i in range(x.ndim):
    diff = shape[i] - x.shape[i]
    assert diff >= 0
    if both:
      a, b = divmod(diff, 2)
      b += a
      pad.append((a, b))
    else:
      pad.append((0, diff))
  pad = tuple(pad)  # type: ignore
  x = np.pad(x, pad, mode='constant', constant_values=fill)
  return x