import inspect
import logging
import numpy as np
import pandas as pd
from typing import Any, Iterable, Union, List, Generator
import os
from jaqpotpy.entities.material_models import Atoms

logger = logging.getLogger(__name__)
_print_threshold = 10

class Parser(object):
  """
  Abstract class for parsing molecules or materials files into serializable objects.
  This class is abstract and cannot be invoked directly. You'll
  likely only interact with this class if you're a developer. In
  that case, you might want to make a child class which
  implements the `_parse` (or the `_parse_dataframe`) method for calculating parsing for
  parsing files representing materials or molecules.
  """

  def __init__(self, path: str, file_ext: Union[str, List[str]]):
    self.path = path
    if self.path[-1] == '/':
      self.path = self.path[:-1]

    self.file_ext = file_ext
    if isinstance(self.file_ext, str):
      self.file_ext = [self.file_ext]

    self.files_ = []

  def parse(self) -> Generator:
    """
    Parse files representing materials or molecules.

    Parameters
    ----------
    path: str
      Either the path of a certain file or a path of a folder containing
      files that will be parsed

    file_ext: str | List[str]
      The extention of the files that will be parsed.

    log_every_n: int, default 1000
      Logs featurization progress every `log_every_n` steps.

    Returns
    -------
    generator
      A generator with the parsed objects of the files.
    """

    try:
      if self.path.split('.')[-1].lower() not in self.file_ext:
        try:
          file = [self.path+'/'+item for item in os.listdir(self.path) if item.split('.')[-1].lower() in self.file_ext]
        except:
          raise ValueError('Invalid file type. Expected {} but found {} instead'.format(self.file_ext, self.path.split('.')[-1]))
      else:
        file = [self.path]
    except:
      try:
        file = [self.path+'/'+item for item in os.listdir(self.path) if item.split('.')[-1].lower() in self.file_ext]
      except:
        raise ValueError("Invalid file type. Expected {} but didn't find any".format(self.file_ext))

    # Open the file and read it in as a list of rows
    for i, item in enumerate(file):
      try:
        obj = self._parse(item)
        if isinstance(obj, list):
          for i in obj:
            yield i
        else:
           yield obj
      except Exception as e:
        logger.warning("Failed to parse file {}.".format(item), e)


  def parse_dataframe(self) -> Union[List[pd.DataFrame], pd.DataFrame]:
    """
    Parse files of materials or molecules in a pandas dataframe.

    Parameters
    ----------
    path: str
      Either the path of a certain file or a path of a folder containing
      files that will be parsed

    file_ext: str | List[str]
      The extention of the files that will be parsed.

    log_every_n: int, default 1000
      Logs featurization progress every `log_every_n` steps.

    Returns
    -------
    pd.DataFrame
      A pandas dataframe of the files.
    """

    files = self.parse()

    df = pd.DataFrame()
    df_list = []
    for i, file in enumerate(files):
      resp = self._parse_dataframe(file, self.files_[i])
      try:
        df = pd.concat([df, resp]).reset_index(drop=True)
      except:
        if len(df_list) == 0:
          df_list = [pd.concat([df, item]).reset_index(drop=True) for item in resp]
        else:
          for j in range(len(df_list)):
            df_list[j] = pd.concat([df_list[j], resp[j]]).reset_index(drop=True)

    if len(df_list)>0:
      return df_list
    return df

  def __call__(self, path: str, file_ext: Union[str, List[str]], **kwargs):
    """Calculate features for datapoints.
    `**kwargs` will get passed directly to `Featurizer.featurize`
    Parameters
    ----------
    datapoints: Iterable[Any]
      Any blob of data you like. Subclasss should instantiate this.
    """
    return self.parse()

  def _parse(self, path: str):
    """Calculate features for a single datapoint.
    Parameters
    ----------
    datapoint: Any
      Any blob of data you like. Subclass should instantiate this.
    """
    raise NotImplementedError('Parser is not defined.')

  def _parse_dataframe(self, file: Any, filename: str) -> Union[List[pd.DataFrame], pd.DataFrame]:
    """Calculate features for a single datapoint.
    Parameters
    ----------
    datapoint: Any
      Any blob of data you like. Subclass should instantiate this.
    """
    raise NotImplementedError('Parser is not defined.')


  def __repr__(self) -> str:
    """Convert self to repr representation.
    Returns
    -------
    str
      The string represents the class.
    Examples
    --------
    >>> import jaqpotpy as jt
    >>> jt.parsers.PdbParser('pdb_file.pdb')
    PdbParser[pdb='pdb_file.pdb', file_ext='pdb']
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
    >>> jt.parsers.PdbParser('pdb_file.pdb')
    PdbParser[pdb='pdb_file.pdb', file_ext='pdb']
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

  def to_xyz(self):
    xyzs = []
    files = self.parse()
    for file in files:
      atoms:Atoms = file.get_atoms()
      xyz = str(len(atoms.elements)) + '\ntemp xyz\n'
      for i in range(len(atoms.elements)):
        xyz += atoms.elements[i] + '\t' + str(atoms.coordinates[i][0]) + '\t' + str(atoms.coordinates[i][1]) + '\t' + str(atoms.coordinates[i][2]) + '\n'

      xyzs.append(xyz)

    return xyzs


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