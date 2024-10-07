"""Dataset abstract classes"""

from abc import ABC, abstractmethod
import os
import pickle
from typing import Iterable, Optional
import pandas as pd


class BaseDataset(ABC):
    """Abstract class for datasets. This class defines the common interface and basic functionality
    for dataset manipulation and handling.

    Attributes
    ----------
        df (pd.DataFrame): The underlying DataFrame holding the dataset.
        x_cols (Optional[Iterable[str]]): The columns to be used as features.
        y_cols (Optional[Iterable[str]]): The columns to be used as labels.
        _task (str): The task type, either 'regression' or 'classification'.
        _dataset_name (str): The name of the dataset.
        y (Iterable[str]): The labels of the dataset.
        X (Iterable[str]): The features of the dataset.

    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        path: Optional[str] = None,
        y_cols: Iterable[str] = None,
        x_cols: Optional[Iterable[str]] = None,
        task: str = None,
    ) -> None:
        if df is None and path is None:
            raise TypeError("Either a DataFrame or a path to a file must be provided.")
        elif (df is not None) and (path is not None):
            raise TypeError("Either a DataFrame or a path to a file must be provided.")

        if df is not None:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Provided 'df' must be a pandas DataFrame.")
            else:
                self.df = df
                self.path = None
        elif path is not None:
            self.path = path
            extension = os.path.splitext(self.path)[1]
            if extension == ".csv":
                self.df = pd.read_csv(path)
            else:
                raise ValueError("The provided file is not a valid CSV file.")

        if not (
            isinstance(y_cols, str)
            or (
                isinstance(y_cols, list)
                and all(isinstance(item, str) for item in y_cols)
            )
            or (y_cols is None)
        ):
            raise TypeError(
                "y_cols must be provided and should be either"
                "a string or a list of strings, or None"
            )

        if not (
            isinstance(x_cols, str)
            or (
                isinstance(x_cols, list)
                and all(isinstance(item, str) for item in x_cols)
            )
            or (isinstance(x_cols, list) and len(x_cols) == 0)
            or (x_cols is None)
        ):
            raise TypeError(
                "x_cols should be either a string, an empty list"
                "a list of strings, or None"
            )

        # Find the length of each provided column name vector and put everything in lists
        if isinstance(y_cols, str):
            self.y_cols = [y_cols]
            self.y_cols_len = 1
        elif isinstance(y_cols, list):
            self.y_cols = y_cols
            self.y_cols_len = len(y_cols)
        elif y_cols is None:
            self.y_cols = []
            self.y_cols_len = 0

        if isinstance(x_cols, str):
            self.x_cols = [x_cols]
            self.x_cols_len = 1
        elif isinstance(x_cols, list):
            self.x_cols = x_cols
            self.x_cols_len = len(x_cols)
        elif x_cols is None:
            self.x_cols = []
            self.x_cols_len = 0

        self.task = task
        self._dataset_name = None
        self.y = None
        self.X = None

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        if value is None or value.upper() not in [
            "REGRESSION",
            "BINARY_CLASSIFICATION",
            "MULTICLASS_CLASSIFICATION",
        ]:
            raise ValueError(
                "Task must be either 'REGRESSION', BINARY_CLASSIFICATION' or 'MULTICLASS_CLASSIFICATION'."
            )

        self._task = value.upper()

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value):
        self._dataset_name = value

    def save(self):
        if self._dataset_name:
            with open(self._dataset_name + ".jdata", "wb") as f:
                pickle.dump(self, f)
        else:
            with open("jaqpot_dataset" + ".jdata", "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    @abstractmethod
    def create(self):
        """Creates the dataset."""
        raise NotImplementedError

    @abstractmethod
    def __get_X__(self):
        """Returns the design matrix X."""
        raise NotImplementedError

    @abstractmethod
    def __get_Y__(self):
        """Returns the response Y."""
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """Returns a string representation of the dataset."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Returns the number of samples in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def __get__(self, instance, owner):
        """Gets an attribute of the dataset."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        """Gets a sample by index."""
        raise NotImplementedError


if __name__ == "__main__":
    ...
