"""Dataset classes for molecular modelling"""

import copy
from typing import Iterable, Optional

import pandas as pd

from jaqpotpy.datasets.dataset_base import BaseDataset
import numpy as np


class JaqpotTensorDataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame = None,
        path: Optional[str] = None,
        y_cols: Iterable[str] = None,
        x_cols: Optional[Iterable[str]] = None,
        task: str = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the JaqpotpyDataset.

        Args:
            df (pd.DataFrame): The DataFrame containing the dataset.
            path (Optional[str]): The path to the dataset file.
            y_cols (Iterable[str]): The columns representing the target variables.
            x_cols (Optional[Iterable[str]]): The columns representing the features.
            task (str): The task type (e.g., regression, classification).
            verbose (bool, optional): If True, enables detailed logging or printing. Default is True.
        """
        super().__init__(
            df=df, path=path, y_cols=y_cols, x_cols=copy.deepcopy(x_cols), task=task
        )
        self._validate_column_names(self.x_cols, "x_cols")
        self._validate_column_names(self.y_cols, "y_cols")
        self._validate_column_space()

        self.init_df = self.df
        self._X_old = None
        self.verbose = verbose
        self.row_index_before_inf_drop = None
        self.create()

    def create(self):
        """
        Create the dataset
        """
        self.df = self.df.reset_index(drop=True)
        self.X = self.df[self.x_cols]

        if not self.y_cols:
            self.y = None
        else:
            self.y = self.df[self.y_cols]
        self.x_colnames = self.X.columns.tolist()
        self.df = pd.concat([self.X, self.y], axis=1)
        self.X.columns = self.X.columns.astype(str)
        self.df.columns = self.df.columns.astype(str)

    def copy(self):
        """
        Create a copy of the dataset, including a deep copy of the underlying DataFrame
        and all relevant attributes.

        Returns:
            JaqpotTabularDataset: A copy of the dataset.
        """
        copied_instance = JaqpotTensorDataset(
            df=self.init_df,
            path=self.path,
            y_cols=self.y_cols,
            x_cols=self.x_cols,
            task=self.task,
        )
        return copied_instance

    def __get_X__(self):
        """
        Get a copy of the feature matrix X.

        Returns:
            pd.DataFrame: A copy of the feature matrix X.
        """
        return self.X.copy()

    def __get_Y__(self):
        """
        Get a copy of the target matrix Y.

        Returns:
            pd.DataFrame: A copy of the target matrix Y.
        """
        return self.y.copy()

    def __get__(self, instance, owner):
        """
        Get the dataset instance.

        Args:
            instance: The instance of the dataset.
            owner: The owner of the dataset.

        Returns:
            The dataset instance.
        """
        if instance is None:
            return self
        return instance.__dict__[self.df]

    def __getitem__(self, idx):
        """
        Get the features and target values for a given index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the features and target values for the given index.
        """
        selected_x = self.X.copy().iloc[idx]
        selected_y = self.y.copy().to_numpy()
        return selected_x, selected_y

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.df.shape[0]

    def __repr__(self) -> str:
        """
        Get the string representation of the dataset.

        Returns:
            str: The string representation of the dataset.
        """
        return f"{self.__class__.__name__}"
