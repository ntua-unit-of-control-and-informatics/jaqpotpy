from jaqpotpy.descriptors.base_classes import MaterialFeaturizer
from pymatgen.core.composition import Composition
from pymatgen.util.typing import CompositionLike
from typing import Any
from jaqpotpy.helpers.statistics import PropertyStats
import numpy as np
import pandas as pd


class ElementPropertyFingerprint(MaterialFeaturizer):
  """
  Fingerprint of elemental properties from composition.
  Based on the data source chosen, returns properties and statistics
  (min, max, range, mean, standard deviation, mode) for a compound
  based on elemental stoichiometry. E.g., the average electronegativity
  of atoms in a crystal structure. The chemical fingerprint is a
  vector of these statistics.
  See references for more details.
  References
  ----------
  .. [1] Pymatgen: Ong, S.P. et al. Comput. Mater. Sci. 68, 314-319 (2013).
  Examples
  --------
  >>> import jaqpotpy as jt
  >>> comp = "Fe2O3"
  >>> featurizer = jt.descriptors.material.ElementPropertyFingerprint()
  >>> features = featurizer.featurize([comp])
  >>> type(features[0])
  <class 'numpy.ndarray'>
  >>> features[0].shape
  (65,)
  Note
  ----
  This class requires matminer and Pymatgen to be installed.
  `NaN` feature values are automatically converted to 0 by this featurizer.
  """

  def __init__(self):
    """
    """
    self.ep_featurizer: Any = None
    self.pstats = PropertyStats()

  @property
  def __name__(self):
    return 'ElementPropertyFingerprint'

  def __getitem__(self):
    return self

  def _get_elemental_property(self, elem, property_name):
      if property_name == "block":
          block_key = {"s": 1.0, "p": 2.0, "d": 3.0, "f": 3.0}
          return block_key[getattr(elem, property_name)]
      else:
          value = getattr(elem, property_name)
          return np.nan if value is None else value

  def _featurize(self, datapoint: CompositionLike, **kwargs) -> np.ndarray:
    """
    Calculate chemical fingerprint from crystal composition.
    Parameters
    ----------
    datapoint: pymatgen.core.Composition object
      Composition object.
    Returns
    -------
    feats: np.ndarray
      Vector of properties and statistics derived from chemical
      stoichiometry. Some values may be NaN.
    """

    stats = ["minimum", "maximum", "range", "mean", "std_dev"]
    features = [
        "X",
        "row",
        "group",
        "block",
        "atomic_mass",
        "atomic_radius",
        "mendeleev_no",
        "electrical_resistivity",
        "velocity_of_sound",
        "thermal_conductivity",
        "melting_point",
        "bulk_modulus",
        "coefficient_of_linear_thermal_expansion",
    ]

    if not isinstance(datapoint, Composition):
        data = Composition(datapoint)
    else:
        data = datapoint

    all_attributes = []

    # Get the element names and fractions
    elements, fractions = zip(*data.element_composition.items())

    for attr in features:
      try:
        elem_data = [self._get_elemental_property(e, attr) for e in elements]
      except:
        raise ValueError('Cannot find property of Element.')

      for stat in stats:
        all_attributes.append(self.pstats.calc_stat(elem_data, stat, fractions))

    return np.array(all_attributes)

  def _featurize_dataframe(self, datapoint: CompositionLike, **kwargs) -> pd.DataFrame:
    """
    Calculate chemical fingerprint from crystal composition.
    Parameters
    ----------
    datapoint: pymatgen.core.Composition object
      Composition object.
    Returns
    -------
    feats: np.ndarray
      Vector of properties and statistics derived from chemical
      stoichiometry. Some values may be NaN.
    """

    stats = ["minimum", "maximum", "range", "mean", "std_dev"]
    features = [
        "X",
        "row",
        "group",
        "block",
        "atomic_mass",
        "atomic_radius",
        "mendeleev_no",
        "electrical_resistivity",
        "velocity_of_sound",
        "thermal_conductivity",
        "melting_point",
        "bulk_modulus",
        "coefficient_of_linear_thermal_expansion",
    ]

    if not isinstance(datapoint, Composition):
        data = Composition(datapoint)
    else:
        data = datapoint

    all_attributes = []

    # Get the element names and fractions
    elements, fractions = zip(*data.element_composition.items())

    for attr in features:
      try:
        elem_data = [self._get_elemental_property(e, attr) for e in elements]
      except:
        raise ValueError('Cannot find property of Element.')

      for stat in stats:
        all_attributes.append(self.pstats.calc_stat(elem_data, stat, fractions))

    columns = [i + '_' + j for j in features for i in stats]
    return pd.DataFrame(index=[0],columns=columns, data=np.array(all_attributes).reshape(1,-1))

