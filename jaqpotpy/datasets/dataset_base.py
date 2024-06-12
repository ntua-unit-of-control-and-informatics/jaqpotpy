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
        """
        Initializes the dataset either from a DataFrame or from a CSV file.

        Args:
            df (pd.DataFrame, optional): The DataFrame containing the dataset.
            path (str, optional): The path to a CSV file containing the dataset.
            y_cols (Iterable[str], optional): The columns to be used as labels.
            x_cols (Iterable[str], optional): The columns to be used as features.

        Raises:
            ValueError: If neither a DataFrame nor a path is provided.
            ValueError: If the provided df is not a pandas DataFrame.
        """
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

        self.x_cols = x_cols
        self.y_cols = y_cols
        self._task = None
        self._dataset_name = None
        self._y = None
        self._x = None

    @property
    def df(self)-> pd.DataFrame:
        """
        Gets the data frame.

        Returns:
            pd.DataFrame: A dataframe containing the data
        """
        return self._df

    @df.setter
    def df(self, value):
        """
        Sets the dataframe.

        Args:
            value (pd.Dataframe):a new pandas dataframe

        """
        self._df = value


    @property
    def task(self):
        """
        Gets the task type.

        Returns:
            str: The task type, either 'regression' or 'classification'.
        """
        return self._task

    @task.setter
    def task(self, value):
        """
        Sets the task type.

        Args:
            value (str): The task type, must be either 'regression' or 'classification'.

        Raises:
            ValueError: If the task type is not 'regression' or 'classification'.
        """
        if value.lower() not in ['regression', 'classification']:
            raise ValueError("Task must be either 'regression' or 'classification'")
        self._task = value

    @property
    def dataset_name(self):
        """
        Gets the dataset name.

        Returns:
            str: The name of the dataset.
        """
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value):
        """
        Sets the dataset name.

        Args:
            value (str): The name of the dataset.
        """
        self._dataset_name = value

    @property
    def x(self) -> Iterable[str]:
        """
        Gets the features of the dataset.

        Returns:
            Iterable[str]: The features of the dataset.
        """
        return self._x

    @x.setter
    def x(self, value):
        """
        Sets the features of the dataset.

        Args:
            value (Iterable[str]): The features of the dataset.
        """
        self._x = value

    @property
    def y(self) -> Iterable[str]:
        """
        Gets the labels of the dataset.

        Returns:
            Iterable[str]: The labels of the dataset.
        """
        return self._y

    @y.setter
    def y(self, value):
        """
        Sets the labels of the dataset.

        Args:
            value (Iterable[str]): The labels of the dataset.
        """
        self._y = value

    def save(self):
        """
        Saves the dataset object to a file.

        The file is saved with the name of the dataset (if provided) or with a default name.
        """
        if self._dataset_name:
            with open(self._dataset_name + ".jdata", 'wb') as f:
                pickle.dump(self, f)
        else:
            with open("jaqpot_dataset" + ".jdata", 'wb') as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """
        Loads a dataset object from a file.

        Args:
            filename (str): The file from which to load the dataset.

        Returns:
            BaseDataset: The loaded dataset object.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @abstractmethod
    def create(self):
        """
        Creates the dataset.

        This method should be implemented by subclasses to define how the dataset is created.
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """
        Returns a string representation of the dataset.

        Returns:
            str: A string representation of the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __get_x__(self):
        """
        Gets the feature matrix.

        This method should be implemented by subclasses to define how the features are accessed.
        """
        raise NotImplementedError

    @abstractmethod
    def __get_y__(self):
        """
        Gets the label array.

        This method should be implemented by subclasses to define how the labels are accessed.
        """
        raise NotImplementedError

    @abstractmethod
    def __get__(self, instance, owner):
        """
        Gets an attribute of the dataset.

        This method should be implemented by subclasses to define how attributes are accessed.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        """
        Gets a sample by index.

        Args:
            idx (int): The index of the sample to get.

        Returns:
            tuple: The (features, labels) of the sample.
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
