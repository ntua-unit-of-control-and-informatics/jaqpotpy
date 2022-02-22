"""
Dataset base classes
"""
from typing import Any
import inspect
from typing import Iterable


class MolecularDataset(object):
    """
    Astract class for datasets
    """
    def __init__(self, path=None, smiles_col=None, x_cols=None, y_cols=None, smiles=None) -> None:
        self._Y = None
        self._X = None
        self._dataset_name = None
        self._df = None
        self._x_cols_all = None
        self.smiles = smiles
        self._smiles_strings = None
        self.path = path
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.smiles_col = smiles_col
        self._task = "regression"

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        self._task = value

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value):
        self._dataset_name = value

    @property
    def featurizer_name(self) -> Iterable[Any]:
        return self.featurizer.__name__

    @property
    def x_colls_all(self) -> Iterable[str]:
        return self._x_cols_all

    @x_colls_all.setter
    def x_colls_all(self, value):
        self._x_cols_all = value

    @property
    def X(self) -> Iterable[str]:
        return self._x

    @X.setter
    def X(self, value):
        self._x = value

    @property
    def y(self) -> Iterable[str]:
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def external(self) -> Iterable[str]:
        return self._external

    @external.setter
    def external(self, value):
        self._external = value

    @property
    def smiles_strings(self) -> Iterable[str]:
        return self._smiles_strings

    @smiles_strings.setter
    def smiles_strings(self, value):
        self._smiles_strings = value

    @property
    def df(self) -> Any:
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    @featurizer_name.setter
    def featurizer_name(self, value):
        self._featurizer_name = value

    def create(self):
        raise NotImplementedError("Need implementation")

    def __repr__(self) -> str:
        args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
        args_names = [arg for arg in args_spec.args if arg != 'self']
        args_info = ''
        for arg_name in args_names:
          value = self.__dict__[arg_name]
          # for str
          if isinstance(value, str):
            value = "'" + value + "'"
          # for list
        return self.__class__.__name__

    # @x_cols.setter
    # def x_cols(self, value):
    #     self._x_cols = value
    #
    # @y_cols.setter
    # def y_cols(self, value):
    #     self._y_cols = value
