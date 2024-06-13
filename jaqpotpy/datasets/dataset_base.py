"""
Dataset abstract classes
"""
from abc import ABC, abstractmethod
import os
import pickle
from typing import Any, Iterable, Optional
import pandas as pd
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer

class BaseDataset(ABC):
    """
    Abstract class for datasets. This class defines the common interface and basic functionality 
    for dataset manipulation and handling.

    Attributes:
        _df (pd.DataFrame): The underlying DataFrame holding the dataset.
        x_cols (Optional[Iterable[str]]): The columns to be used as features.
        y_cols (Optional[Iterable[str]]): The columns to be used as labels.
        _task (str): The task type, either 'regression' or 'classification'.
        _dataset_name (str): The name of the dataset.
        _y (Iterable[str]): The labels of the dataset.
        _x (Iterable[str]): The features of the dataset.
    """

    def __init__(self, df: pd.DataFrame = None, path: Optional[str] = None,
                 y_cols: Iterable[str] = None,
                 x_cols: Optional[Iterable[str]] = None) -> None:

        if df is None and path is None:
            raise ValueError("Either a DataFrame or a path to a file must be provided.")

        if df is not None:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Provided 'df' must be a pandas DataFrame.")
            else:
                self._df = df
                self.path = None
        elif path is not None:
            self.path = path
            name, extension = os.path.splitext(self.path)
            if extension == '.csv':
                self._df = pd.read_csv(path)
            else:
                raise ValueError("The provided file is not a valid CSV file.")
        
        if isinstance(y_cols, str):
            self.y_cols = y_cols
            self.y_cols_len = 1
        elif isinstance(y_cols, list) and all(isinstance(item, str) for item in y_cols):
            self.y_cols = y_cols
            self.y_cols_len = len(y_cols)
        else:
            raise TypeError("y_cols must be a string or a list of strings.")
        
        if isinstance(x_cols, str):
            self.x_cols = y_cols
            self.x_cols_len = 1
        elif isinstance(x_cols, list) and all(isinstance(item, str) for item in x_cols):
            self.x_cols = x_cols
            self.x_cols_len = len(x_cols)
        elif x_cols is None:
            self.x_cols = None
            self.x_cols_len = 0
        else:
            raise TypeError("x_cols must either be a string, a list of strings or a None type.")

        self._task = None
        self._dataset_name = None
        self._y = None
        self._x = None

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("The value must be a pandas DataFrame.")
        self._df = value

    @df.deleter
    def df(self):
        del self._df

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        if value.lower() not in ['regression', 'classification']:
            raise ValueError("Task must be either 'regression' or 'classification'")
        self._task = value

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value):
        self._dataset_name = value

    @property
    def x(self) -> Iterable[str]:
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self) -> Iterable[str]:
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    def save(self):
        if self._dataset_name:
            with open(self._dataset_name + ".jdata", 'wb') as f:
                pickle.dump(self, f)
        else:
            with open("jaqpot_dataset" + ".jdata", 'wb') as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @abstractmethod
    def create(self):
        """
        Creates the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """
        Returns a string representation of the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __get_x__(self):
        """
        Gets the feature matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def __get_y__(self):
        """
        Gets the label array.
        """
        raise NotImplementedError

    @abstractmethod
    def __get__(self, instance, owner):
        """
        Gets an attribute of the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        """
        Gets a sample by index.
        """
        raise NotImplementedError


class MaterialDataset(BaseDataset):
    def __init__(self, df: pd.DataFrame = None, path:  Optional[str] = None,
                 y_cols: Iterable[str] = None,
                 x_cols: Optional[Iterable[str]] =None, materials_col=None,
                 materials=None) -> None:
        super().__init__(df = df, path = path, y_cols = y_cols, x_cols = x_cols)
        self.materials = materials
        self._materials_strings = None
        self.materials_col = materials_col

    @property
    def materials_strings(self) -> Iterable[str]:
        return self._materials_strings

    @materials_strings.setter
    def materials_strings(self, value):
        self._materials_strings = value


class ImageDataset(BaseDataset):
    def __init__(self,  df: pd.DataFrame = None, path:  Optional[str] = None,
                  y_cols: Iterable[str] = None,
                 x_cols: Optional[Iterable[str]] =None) -> None:
        super().__init__(df = df, path = path, y_cols = y_cols, x_cols = x_cols)


if __name__ == '__main__':
    ...
