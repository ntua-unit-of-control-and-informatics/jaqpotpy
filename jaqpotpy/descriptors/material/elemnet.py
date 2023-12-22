from jaqpotpy.descriptors.base_classes import MaterialFeaturizer
from pymatgen.util.typing import CompositionLike, Composition
import numpy as np
import pandas as pd

class ElementNet(MaterialFeaturizer):
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
  >>> comp = "Fe2O3"
  >>> featurizer = jt.descriptors.material.ElementNet()
  >>> features = featurizer.featurize([comp])
  >>> type(features[0])
  <class 'numpy.ndarray'>
  >>> features[0].shape
  (118,)
  Note
  ----
  This class requires matminer and Pymatgen to be installed.
  `NaN` feature values are automatically converted to 0 by this featurizer.
  """

  def __init__(self):
    """
   Create a fixed vector of length 118, containing rwa, fractional
    element compostitions in a compound.

      References
      ----------
      Jha, D., Ward, L., Paul, A. et al. Sci Rep 8, 17593 (2018).
         https://doi.org/10.1038/s41598-018-35934-y
    """
    self.elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
       'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
       'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
       'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
       'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
       'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
       'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
       'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
       'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
       'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
       'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    return

  @property
  def __name__(self):
    return 'ElementNet'

  def __getitem__(self):
    return self


  def _featurize(self, datapoint: CompositionLike, **kwargs) -> np.ndarray:
    """
    Calculate 118 dimensional vector containing fractional
    compositions of each element in the compound.

    Parameters
    ----------
    datapoint: pymatgen.utils.typing.CompositionLike object
      Composition or an object that can be used to creaet a composition

    Returns
    -------
    feats: np.ndarray
      Vector containing the fractional compositions of each element.
    """
    if not isinstance(datapoint, Composition):
        datapoint = Composition(datapoint)
    fractions = datapoint.fractional_composition.get_el_amt_dict()
    return np.array([fractions[e] if e in fractions else 0 for e in self.elements])

  def _featurize_dataframe(self, datapoint: CompositionLike, **kwargs) -> pd.DataFrame:
    """
    Calculate 118 dimensional vector containing fractional
    compositions of each element in the compound.

    Parameters
    ----------
    datapoint: pymatgen.utils.typing.CompositionLike object
      Composition or an object that can be used to create a composition

    Returns
    -------
    feats: pd.Dataframe
      Dataframe of the vector containing the fractional compositions of each element.
    """

    data = self._featurize(datapoint)
    return pd.DataFrame([data], columns=self.elements, index=[0])

