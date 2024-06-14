import inspect
import logging
import numpy as np
from typing import Any, List, Iterable, Optional, Tuple, Union, cast
import pandas as pd
from tqdm import tqdm
import os
from jaqpotpy.cfg import config

logger = logging.getLogger(__name__)
_print_threshold = 10


class Featurizer(object):
  """Abstract class for calculating a set of features for a datapoint.
  This class is abstract and cannot be invoked directly. You'll
  likely only interact with this class if you're a developer. In
  that case, you might want to make a child class which
  implements the `_featurize` method for calculating features for
  a single datapoints if you'd like to make a featurizer for a
  new datatype.
    References
  ----------
  .. [1] Ramsundar-et-al-2019,. "Deep Learning for the Life Sciences."
     O'Reilly Media (2019):.
  .. [2] https://github.com/deepchem/deepchem
  """

  def featurize(self,
                datapoints: Iterable[Any],
                log_every_n: int = 1000,
                **kwargs) -> np.ndarray:
    """Calculate features for datapoints.
    Parameters
    ----------
    datapoints: Iterable[Any]
      A sequence of objects that you'd like to featurize. Subclassses of
      `Featurizer` should instantiate the `_featurize` method that featurizes
      objects in the sequence.
    log_every_n: int, default 1000
      Logs featurization progress every `log_every_n` steps.
    Returns
    -------
    np.ndarray
      A numpy array containing a featurized representation of `datapoints`.
    """
    datapoints = list(datapoints)
    features = []
    if config.verbose is False:
      disable_tq = True
    else:
      disable_tq = False
    for i, point in enumerate(tqdm(datapoints, desc='Creating descriptors', disable=disable_tq)):

      try:
        features.append(self._featurize(point, **kwargs))
      except:
        if config.verbose is True:
          logger.warning(
              "Failed to featurize datapoint %d. Appending empty array")
        features.append(np.array([]))

    return np.asarray(features)

  def __call__(self, datapoints: Iterable[Any], **kwargs):
    """Calculate features for datapoints.
    `**kwargs` will get passed directly to `Featurizer.featurize`
    Parameters
    ----------
    datapoints: Iterable[Any]
      Any blob of data you like. Subclasss should instantiate this.
    """
    return self.featurize(datapoints, **kwargs)

  def _featurize(self, datapoint: Any, **kwargs):
    """Calculate features for a single datapoint.
    Parameters
    ----------
    datapoint: Any
      Any blob of data you like. Subclass should instantiate this.
    """
    raise NotImplementedError('Featurizer is not defined.')

  def _featurize_dataframe(self, datapoint: Any, **kwargs):
    """Calculate features for a single datapoint.
    Parameters
    ----------
    datapoint: Any
      Any blob of data you like. Subclass should instantiate this.
    """
    raise NotImplementedError('Featurizer is not defined.')

  def __repr__(self) -> str:
    """Convert self to repr representation.
    Returns
    -------
    str
      The string represents the class.
    Examples
    --------
    >>> import jaqpotpy as jt
    >>> jt.descriptors.CircularFingerprint(size=1024, radius=4)
    CircularFingerprint[radius=4, size=1024, chiral=False, bonds=True, features=False, sparse=False, smiles=False]
    >>> jt.descriptors.CGCNNFeaturizer()
    CGCNNFeaturizer[radius=8.0, max_neighbors=12, step=0.2]
    """
    args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
    args_names = [arg for arg in args_spec.args if arg != 'self']
    args_info = ''
    for arg_name in args_names:
      value = self.__dict__[arg_name]
      # for str
      if isinstance(value, str):
        value = "'" + value + "'"
      # for list
      if isinstance(value, list):
        threshold = get_print_threshold()
        value = np.array2string(np.array(value), threshold=threshold)
      args_info += arg_name + '=' + str(value) + ', '
    return self.__class__.__name__ + '[' + args_info[:-2] + ']'

  def __str__(self) -> str:
    """Convert self to str representation.
    Returns
    -------
    str
      The string represents the class.
    Examples
    --------
    >>> import jaqpotpy as jt
    >>> jt.descriptors.CircularFingerprint(size=1024, radius=4)
    CircularFingerprint[radius=4, size=1024, chiral=False, bonds=True, features=False, sparse=False, smiles=False]
    >>> jt.descriptors.CGCNNFeaturizer()
    CGCNNFeaturizer[radius=8.0, max_neighbors=12, step=0.2]
    """
    args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
    args_names = [arg for arg in args_spec.args if arg != 'self']
    args_num = len(args_names)
    args_default_values = [None for _ in range(args_num)]
    if args_spec.defaults is not None:
      defaults = list(args_spec.defaults)
      args_default_values[-len(defaults):] = defaults

    override_args_info = ''
    for arg_name, default in zip(args_names, args_default_values):
      if arg_name in self.__dict__:
        arg_value = self.__dict__[arg_name]
        # validation
        # skip list
        if isinstance(arg_value, list):
          continue
        if isinstance(arg_value, str):
          # skip path string
          if "\\/." in arg_value or "/" in arg_value or '.' in arg_value:
            continue
        # main logic
        if default != arg_value:
          override_args_info += '_' + arg_name + '_' + str(arg_value)
    return self.__class__.__name__ + override_args_info

class ComplexFeaturizer(Featurizer):
  """"
  Abstract class for calculating features for mol/protein complexes.
  """

  def featurize(self,
                datapoints: Optional[Iterable[Tuple[str, str]]] = None,
                log_every_n: int = 100,
                **kwargs) -> np.ndarray:
    """
    Calculate features for mol/protein complexes.
    Parameters
    ----------
    datapoints: Iterable[Tuple[str, str]]
      List of filenames (PDB, SDF, etc.) for ligand molecules and proteins.
      Each element should be a tuple of the form (ligand_filename,
      protein_filename).
    Returns
    -------
    features: np.ndarray
      Array of features
    """

    if 'complexes' in kwargs:
      datapoints = kwargs.get("complexes")
      raise DeprecationWarning(
          'Complexes is being phased out as a parameter, please pass "datapoints" instead.'
      )
    if not isinstance(datapoints, Iterable):
      datapoints = [cast(Tuple[str, str], datapoints)]
    features, failures, successes = [], [], []
    for idx, point in enumerate(datapoints):
      if idx % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % idx)
      try:
        features.append(self._featurize(point, **kwargs))
        successes.append(idx)
      except:
        logger.warning(
            "Failed to featurize datapoint %i. Appending empty array." % idx)
        features.append(np.zeros(1))
        failures.append(idx)

    # Find a successful featurization
    try:
      i = np.argmax([f.shape[0] for f in features])
      dtype = features[i].dtype
      shape = features[i].shape
      dummy_array = np.zeros(shape, dtype=dtype)
    except AttributeError:
      dummy_array = features[successes[0]]

    # Replace failed featurizations with appropriate array
    for idx in failures:
      features[idx] = dummy_array

    return np.asarray(features)

  def _featurize(self, datapoint: Optional[Tuple[str, str]] = None, **kwargs):
    """
    Calculate features for single mol/protein complex.
    Parameters
    ----------
    complex: Tuple[str, str]
      Filenames for molecule and protein.
    """
    raise NotImplementedError('Featurizer is not defined.')


class DataframeFeaturizer(object):
  """Abstract class for calculating a set of features for a datapoint.
  This class is abstract and cannot be invoked directly. You'll
  likely only interact with this class if you're a developer. In
  that case, you might want to make a child class which
  implements the `_featurize` method for calculating features for
  a single datapoints if you'd like to make a featurizer for a
  new datatype.
    References
  ----------
  .. [1] Ramsundar-et-al-2019,. "Deep Learning for the Life Sciences."
     O'Reilly Media (2019):.
  .. [2] https://github.com/deepchem/deepchem
  """

  def featurize(self,
                datapoints: Iterable[Any],
                log_every_n: int = 1000,
                **kwargs) -> Any:
    """Calculate features for datapoints.
    Parameters
    ----------
    datapoints: Iterable[Any]
      A sequence of objects that you'd like to featurize. Subclassses of
      `Featurizer` should instantiate the `_featurize` method that featurizes
      objects in the sequence.
    log_every_n: int, default 1000
      Logs featurization progress every `log_every_n` steps.
    Returns
    -------
    np.ndarray
      A numpy array containing a featurized representation of `datapoints`.
    """
    datapoints = list(datapoints)
    features = []
    if config.verbose is False:
      disable_tq = True
    else:
      disable_tq = False
    for i, point in enumerate(tqdm(datapoints, desc='Creating descriptors', disable=disable_tq)):
      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)
      try:
        features.append(self._featurize(point, **kwargs))
      except:
        if config.verbose is True:
          logger.warning(
              "Failed to featurize datapoint %d. Appending empty array")
        features.append(np.array([]))

    return np.asarray(features)

  def __call__(self, datapoints: Iterable[Any], **kwargs):
    """Calculate features for datapoints.
    `**kwargs` will get passed directly to `Featurizer.featurize`
    Parameters
    ----------
    datapoints: Iterable[Any]
      Any blob of data you like. Subclasss should instantiate this.
    """
    return self.featurize(datapoints, **kwargs)

  def _featurize(self, datapoint: Any, **kwargs):
    """Calculate features for a single datapoint.
    Parameters
    ----------
    datapoint: Any
      Any blob of data you like. Subclass should instantiate this.
    """
    raise NotImplementedError('Featurizer is not defined.')

  def __repr__(self) -> str:
    """Convert self to repr representation.
    Returns
    -------
    str
      The string represents the class.
    Examples
    --------
    >>> import jaqpotpy as jt
    >>> jt.descriptors.CircularFingerprint(size=1024, radius=4)
    CircularFingerprint[radius=4, size=1024, chiral=False, bonds=True, features=False, sparse=False, smiles=False]
    >>> jt.descriptors.CGCNNFeaturizer()
    CGCNNFeaturizer[radius=8.0, max_neighbors=12, step=0.2]
    """
    args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
    args_names = [arg for arg in args_spec.args if arg != 'self']
    args_info = ''
    for arg_name in args_names:
      value = self.__dict__[arg_name]
      # for str
      if isinstance(value, str):
        value = "'" + value + "'"
      # for list
      if isinstance(value, list):
        threshold = get_print_threshold()
        value = np.array2string(np.array(value), threshold=threshold)
      args_info += arg_name + '=' + str(value) + ', '
    return self.__class__.__name__ + '[' + args_info[:-2] + ']'

  def __str__(self) -> str:
    """Convert self to str representation.
    Returns
    -------
    str
      The string represents the class.
    Examples
    --------
    >>> import jaqpotpy as jt
    >>> jt.descriptors.CircularFingerprint(size=1024, radius=4)
    CircularFingerprint[radius=4, size=1024, chiral=False, bonds=True, features=False, sparse=False, smiles=False]
    >>> jt.descriptors.CGCNNFeaturizer()
    CGCNNFeaturizer[radius=8.0, max_neighbors=12, step=0.2]
    """
    args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
    args_names = [arg for arg in args_spec.args if arg != 'self']
    args_num = len(args_names)
    args_default_values = [None for _ in range(args_num)]
    if args_spec.defaults is not None:
      defaults = list(args_spec.defaults)
      args_default_values[-len(defaults):] = defaults

    override_args_info = ''
    for arg_name, default in zip(args_names, args_default_values):
      if arg_name in self.__dict__:
        arg_value = self.__dict__[arg_name]
        # validation
        # skip list
        if isinstance(arg_value, list):
          continue
        if isinstance(arg_value, str):
          # skip path string
          if "\\/." in arg_value or "/" in arg_value or '.' in arg_value:
            continue
        # main logic
        if default != arg_value:
          override_args_info += '_' + arg_name + '_' + str(arg_value)
    return self.__class__.__name__ + override_args_info


class MolecularFeaturizer(Featurizer):
  """Abstract class for calculating a set of features for a
  molecule.
  The defining feature of a `MolecularFeaturizer` is that it
  uses SMILES strings and RDKit molecule objects to represent
  small molecules. All other featurizers which are subclasses of
  this class should plan to process input which comes as smiles
  strings or RDKit molecules.
  Child classes need to implement the _featurize method for
  calculating features for a single molecule.

  .. [1] Ramsundar-et-al-2019,. "Deep Learning for the Life Sciences."
     O'Reilly Media (2019):.
  .. [2] https://github.com/deepchem/deepchem

  Note
  ----
  The subclasses of this class require RDKit to be installed.
  """
  def featurize(self, datapoints, log_every_n=1000, **kwargs) -> np.ndarray:
    """Calculate features for molecules.
    Parameters
    ----------
    datapoints: rdkit.Chem.rdchem.Mol / SMILES string / iterable
      RDKit Mol, or SMILES string or iterable sequence of RDKit mols/SMILES
      strings.
    log_every_n: int, default 1000
      Logging messages reported every `log_every_n` samples.
    Returns
    -------
    features: np.ndarray
      A numpy array containing a featurized representation of `datapoints`.
    """
    try:
      from rdkit import Chem
      from rdkit.Chem import rdmolfiles
      from rdkit.Chem import rdmolops
      from rdkit.Chem.rdchem import Mol
    except ModuleNotFoundError:
      raise ImportError("This class requires RDKit to be installed.")

    if 'molecules' in kwargs:
      datapoints = kwargs.get("molecules")
      raise DeprecationWarning(
          'Molecules is being phased out as a parameter, please pass "datapoints" instead.'
      )

    # Special case handling of single molecule
    if isinstance(datapoints, str) or isinstance(datapoints, Mol):
      datapoints = [datapoints]
    else:
      # Convert iterables to list
      datapoints = list(datapoints)

    features: list = []
    if config.verbose is False:
      disable_tq = True
    else:
      disable_tq = False
    for i, mol in enumerate(tqdm(datapoints, desc='Creating descriptors', disable=disable_tq)):
      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)

      try:
        if isinstance(mol, str):
          # mol must be a RDKit Mol object, so parse a SMILES
          mol = Chem.MolFromSmiles(mol)
          # SMILES is unique, so set a canonical order of atoms
          new_order = rdmolfiles.CanonicalRankAtoms(mol)
          mol = rdmolops.RenumberAtoms(mol, new_order)

        features.append(self._featurize(mol, **kwargs))
      except Exception as e:
        if isinstance(mol, Chem.rdchem.Mol):
          mol = Chem.MolToSmiles(mol)
        if config.verbose is True:
          logger.warning(
              "Failed to featurize datapoint %d, %s. Appending empty array", i,
              mol)
          logger.warning("Exception message: {}".format(e))
        features.append(np.array([]))

    return np.asarray(features)

  def featurize_dataframe(self, datapoints, log_every_n=1000, **kwargs) -> Any:
    """Calculate features for molecules.
    Parameters
    ----------
    datapoints: rdkit.Chem.rdchem.Mol / SMILES string / iterable
      RDKit Mol, or SMILES string or iterable sequence of RDKit mols/SMILES
      strings.
    log_every_n: int, default 1000
      Logging messages reported every `log_every_n` samples.
    Returns
    -------
    features: pd.Dataframe()
      A pandas Dataframe containing a featurized representation of `datapoints`.
    """
    try:
      from rdkit import Chem
      from rdkit.Chem import rdmolfiles
      from rdkit.Chem import rdmolops
      from rdkit.Chem.rdchem import Mol
    except ModuleNotFoundError:
      raise ImportError("This class requires RDKit to be installed.")

    if 'molecules' in kwargs:
      datapoints = kwargs.get("molecules")
      raise DeprecationWarning(
          'Molecules is being phased out as a parameter, please pass "datapoints" instead.'
      )

    # Special case handling of single molecule
    if isinstance(datapoints, str) or isinstance(datapoints, Mol):
      datapoints = [datapoints]
    else:
      # Convert iterables to list
      datapoints = list(datapoints)

    features: list = []
    if config.verbose is False:
      disable_tq = True
    else:
      disable_tq = False
    for i, mol in enumerate(tqdm(datapoints, desc='Creating descriptors', disable=disable_tq)):

      try:
        if isinstance(mol, str):
          # mol must be a RDKit Mol object, so parse a SMILES
          mol = Chem.MolFromSmiles(mol)
          # SMILES is unique, so set a canonical order of atoms
          new_order = rdmolfiles.CanonicalRankAtoms(mol)
          mol = rdmolops.RenumberAtoms(mol, new_order)
          ar = self._featurize_dataframe(mol, **kwargs)
        else:
          ar = self._featurize_dataframe(mol, **kwargs)
        features.append(ar)
      except Exception as e:
        if isinstance(mol, Chem.rdchem.Mol):
          mol = Chem.MolToSmiles(mol)
        if config.verbose is True:
          logger.warning(
              "Failed to featurize datapoint %d, %s. Appending empty array", i,
              mol)
          logger.warning("Exception message: {}".format(e))
        features.append(np.array([]))
        # features.append(self._featurize_dataframe(mol, **kwargs))
    columns = self._get_column_names()
    # features = np.array(features)
    if columns == ['Sequence']:
      df = pd.DataFrame({'Sequence': features})
    elif columns == ['OneHotSequence']:
      df = pd.DataFrame({'OneHotSequence': features})
    elif columns == ['SmilesImage']:
      df = pd.DataFrame({'SmilesImage': features})
    elif columns == ['MACCSFingerprint']:
      df = pd.DataFrame({'MACCSFingerprint': features})
    elif columns == ['MolGanGraphs']:
      df = pd.DataFrame({'MolGanGraphs': features})
    elif columns == ['CoulombMatrixEig']:
      df = pd.DataFrame({'CoulombMatrixEig': features})
    else:
      df = pd.DataFrame(features, columns=columns)
    return df


class ParserFeaturizer(Featurizer):
  """Abstract class for calculating a set of features for one or
  a set of files. The defining feature of a `ParserFeaturizer`
  is that it uses files to generate descriptors. All other
  featurizers which are subclasses of this class
  should plan to process input which comes as a parser of a file or
  a folder with files. Child classes need to implement the
  _featurize method for calculating features for a single instance.
  """

  def featurize(self, log_every_n=1000, **kwargs) -> np.ndarray:
    """Calculate features for materials.
    Parameters
    ----------
    material: string
      Either the path of a pdb file or the path to the folder with many pdb files.
    log_every_n: int, default 1000
      Logging messages reported every `log_every_n` samples.
    files: str
      Extention of the files - Type of representation of materials
    Returns
    -------
    features: np.ndarray
      A numpy array containing a featurized representation of `datapoints`.
    """

    features: list = []
    if config.verbose is False:
      disable_tq = True
    else:
      disable_tq = False
    for i, mat in enumerate(tqdm(self.parser.parse(), desc='Creating descriptors', disable=disable_tq)):
      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)

      try:
        features.append(self._featurize(mat, **kwargs))

      except Exception as e:
        if config.verbose is True:
          logger.warning(
              "Failed to featurize datapoint %d, %s. Appending empty array", i, mat)
          logger.warning("Exception message: {}".format(e))
        features.append(np.array([]))
    return np.asarray(features)

  def featurize_dataframe(self, log_every_n=1000, **kwargs) -> Any:
    """Calculate features for materials.
    Parameters
    ----------
    pdb: string
      Either the path of a pdb file or the path to the folder with many pdb files.
    log_every_n: int, default 1000
      Logging messages reported every `log_every_n` samples.
    files: str
      Extention of the files - Type of representation of materials.

    Returns
    -------
    features: pd.Dataframe()
      A pandas Dataframe containing a featurized representation of `datapoints`.
    """

    features = pd.DataFrame()
    if config.verbose is False:
      disable_tq = True
    else:
      disable_tq = False
    for i, mat in enumerate(tqdm(self.parser.parse(), desc='Creating descriptors', disable=disable_tq)):
      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)

      try:
        features = pd.concat([features, self._featurize_dataframe(mat, **kwargs)]).reset_index(drop=True).replace(np.nan, 0)

      except Exception as e:
        if config.verbose is True:
          logger.warning(
            "Failed to featurize datapoint %d, %s. Appending empty array", i)
          logger.warning("Exception message: {}".format(e))
        features = pd.concat([features, pd.DataFrame()]).reset_index(drop=True)

    return features


class MaterialFeaturizer(Featurizer):
  """
  Abstract class for calculating a set of features for an
  inorganic crystal composition.
  The defining feature of a `MaterialFeaturizer` is that it
  operates on 3D crystal materials.
  Inorganic crystal compositions are represented by Pymatgen composition
  objects. Featurizers for inorganic crystal compositions that are
  subclasses of this class should plan to process input which comes as
  Pymatgen composition objects.
  This class is abstract and cannot be invoked directly. You'll
  likely only interact with this class if you're a developer. Child
  classes need to implement the _featurize method for calculating
  features for a single crystal composition.
  Note
  ----
  Some subclasses of this class will require pymatgen and matminer to be
  installed.
  """

  def featurize(self,
                datapoints: Union[List, Iterable[str]],
                log_every_n: int = 1000,
                **kwargs) -> np.ndarray:
    """Calculate features for crystal compositions.
    Parameters
    ----------
    datapoints: Iterable[str]
      Iterable sequence of composition strings, e.g. "MoS2".
    log_every_n: int, default 1000
      Logging messages reported every `log_every_n` samples.
    Returns
    -------
    features: np.ndarray
      A numpy array containing a featurized representation of
      `compositions`.
    """

    if not isinstance(datapoints, List):
      datapoints = [datapoints]

    if config.verbose is False:
      disable_tq = True
    else:
      disable_tq = False

    features = []
    for i, item in enumerate(tqdm(datapoints, desc='Creating descriptors', disable=disable_tq)):
      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)

      try:
        features.append(self._featurize(item, **kwargs))
      except Exception as e:
        if config.verbose is True:
          logger.warning(
              "Failed to featurize datapoint %i. Appending empty array" % i, e)
        features.append(np.array([]))

    return np.asarray(features)

  def featurize_dataframe(self, datapoints: Union[List, Iterable[str]], log_every_n=1000, **kwargs) -> pd.DataFrame:
    """Calculate features for materials.
    Parameters
    ----------
    pdb: string
      Either the path of a pdb file or the path to the folder with many pdb files.
    log_every_n: int, default 1000
      Logging messages reported every `log_every_n` samples.
    files: str
      Extention of the files - Type of representation of materials.

    Returns
    -------
    features: pd.Dataframe()
      A pandas Dataframe containing a featurized representation of `datapoints`.
    """

    if not isinstance(datapoints, List):
      datapoints = [datapoints]

    if config.verbose is False:
      disable_tq = True
    else:
      disable_tq = False

    datapoints = list(datapoints)
    features = pd.DataFrame()
    for i, item in enumerate(tqdm(datapoints, desc='Creating descriptors', disable=disable_tq)):
      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)

      try:
        features = pd.concat([features, self._featurize_dataframe(item, **kwargs)]).reset_index(drop=True)

      except Exception as e:
        if config.verbose is True:
          logger.warning(
            "Failed to featurize datapoint %d, %s. Appending empty array", i)
          logger.warning("Exception message: {}".format(e))
        features = pd.concat([features, pd.DataFrame()]).reset_index(drop=True)

    return features


def get_print_threshold() -> int:
  """Return the printing threshold for datasets.
  The print threshold is the number of elements from ids/tasks to
  print when printing representations of `Dataset` objects.
  Returns
  ----------
  threshold: int
    Number of elements that will be printed
  """
  return _print_threshold