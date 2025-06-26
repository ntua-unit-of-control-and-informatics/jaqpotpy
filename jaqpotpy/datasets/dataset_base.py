"""Dataset abstract classes"""

from abc import ABC, abstractmethod
import os
import pickle
from typing import Iterable, Optional
import pandas as pd
from jaqpot_api_client.models.model_task import ModelTask


class BaseDataset(ABC):
    """Abstract class for datasets. This class defines the common interface and basic functionality
    for dataset manipulation and handling.

    Attributes
    ----------
    df : pd.DataFrame
        The underlying DataFrame holding the dataset.
    x_cols : Optional[Iterable[str]]
        The columns to be used as features.
    y_cols : Optional[Iterable[str]]
        The columns to be used as labels.
    _task : ModelTask
        The task type (ModelTask.REGRESSION, ModelTask.BINARY_CLASSIFICATION, or ModelTask.MULTICLASS_CLASSIFICATION).
    _dataset_name : str
        The name of the dataset.
    y : Iterable[str]
        The labels of the dataset.
    X : Iterable[str]
        The features of the dataset.
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        path: Optional[str] = None,
        y_cols: Iterable[str] = None,
        x_cols: Optional[Iterable[str]] = None,
        task: ModelTask = None,
    ) -> None:
        """
        Initializes the BaseDataset with either a DataFrame or a path to a CSV file.

        Parameters
        ----------
        df : pd.DataFrame, optional
            The DataFrame containing the dataset.
        path : str, optional
            The path to a CSV file containing the dataset.
        y_cols : Iterable[str], optional
            The columns to be used as labels.
        x_cols : Iterable[str], optional
            The columns to be used as features.
        task : ModelTask, optional
            The task type (ModelTask.REGRESSION, ModelTask.BINARY_CLASSIFICATION, or ModelTask.MULTICLASS_CLASSIFICATION).

        Raises
        ------
        TypeError
            If both df and path are provided or neither is provided.
        ValueError
            If the provided file is not a valid CSV file.
        """
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
        """
        Gets the task type.

        Returns
        -------
        ModelTask
            The task type.
        """
        return self._task

    @task.setter
    def task(self, value):
        """
        Sets the task type.

        Parameters
        ----------
        value : ModelTask
            The task type (ModelTask.REGRESSION, ModelTask.BINARY_CLASSIFICATION, or ModelTask.MULTICLASS_CLASSIFICATION).

        Raises
        ------
        ValueError
            If the task type is not one of the allowed values.
        """
        if value is None or value not in [
            ModelTask.REGRESSION,
            ModelTask.BINARY_CLASSIFICATION,
            ModelTask.MULTICLASS_CLASSIFICATION,
        ]:
            raise ValueError(
                "Task must be either ModelTask.REGRESSION, ModelTask.BINARY_CLASSIFICATION, or ModelTask.MULTICLASS_CLASSIFICATION."
            )

        self._task = value

    @property
    def dataset_name(self):
        """
        Gets the dataset name.

        Returns
        -------
        str
            The dataset name.
        """
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value):
        """
        Sets the dataset name.

        Parameters
        ----------
        value : str
            The name of the dataset.
        """
        self._dataset_name = value

    def save(self):
        """
        Saves the dataset to a file using pickle.

        The dataset is saved with the name specified in _dataset_name attribute.
        If _dataset_name is not set, it defaults to 'jaqpot_dataset.jdata'.
        """
        if self._dataset_name:
            with open(self._dataset_name + ".jdata", "wb") as f:
                pickle.dump(self, f)
        else:
            with open("jaqpot_dataset" + ".jdata", "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """
        Loads a dataset from a file using pickle.

        Parameters
        ----------
        filename : str
            The path to the file from which to load the dataset.

        Returns
        -------
        BaseDataset
            The loaded dataset.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    @abstractmethod
    def create(self):
        """
        Creates the dataset.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def __get_X__(self):
        """
        Returns the design matrix X.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def __get_Y__(self):
        """
        Returns the response Y.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """
        Returns a string representation of the dataset.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """
        Returns the number of samples in the dataset.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def __get__(self, instance, owner):
        """
        Gets an attribute of the dataset.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        """
        Gets a sample by index.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def _validate_column_names(self, cols, col_type):
        """
        Validate if the columns specified in cols are present in the DataFrame.

        Args:
            cols: The columns to validate.
            col_type: The type of columns (e.g., 'smiles_cols', 'x_cols', 'y_cols').

        Raises:
            ValueError: If any columns are missing from the DataFrame.
        """
        if len(cols) == 0:
            return

        missing_cols = [col for col in cols if col not in self.df.columns]

        if missing_cols:
            raise ValueError(
                f"The following columns in {col_type} are not present in the DataFrame: {missing_cols}"
            )

    def _validate_column_space(self):
        """
        Validate and replace spaces in column names with underscores.
        """
        for ix, col in enumerate(self.x_cols):
            if " " in col:
                new_col = col.replace(" ", "_")
                self.x_cols[ix] = new_col
                self.df.rename(columns={col: new_col}, inplace=True)
                print(
                    f"Warning: Column names cannot have spaces. Column '{col}' has been renamed to '{new_col}'"
                )

    def df_astype(self, dtype, columns=None):
        """
        Convert the dataset to the specified data type.

        Args:
            dtype: The data type to convert to.
            columns: The columns of the dataset

        Returns:
            JaqpotTabularDataset: The dataset converted to the specified data type.
        """
        if columns is None:
            columns = self.df.columns
        self.df[columns] = self.df[columns].astype(dtype)
        valid_x_columns = [col for col in columns if col in self.X.columns]
        self.X[valid_x_columns] = self.X[valid_x_columns].astype(dtype)
        if self.y_cols in columns:
            self.y = self.y.astype(dtype)
        return self


if __name__ == "__main__":
    ...
