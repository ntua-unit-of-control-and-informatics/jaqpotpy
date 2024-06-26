"""
Author: Ioannis Pitoskas
Contact: jpitoskas@gmail.com
"""

from torch.utils.data import Dataset
import torch
import pandas as pd


class TabularDataset(Dataset):
    """
    A PyTorch Dataset class for handling tabular data.

    Attributes:
        X (torch.tensor): A 2D tensor containing the feature data.
        y (torch.tensor): A 1D tensor containing the target data.
    """

    def __init__(self, X, y=None):
        """
        Initializes the TabularDataset object.

        Args:
            X (numpy.ndarray or pandas.DataFrame): Feature data matrix of shape (n_samples, n_features).
            y (numpy.ndarray or pandas.DataFrame, optional): Target data of shape (n_samples,).
        
        Example:
        ```
        >>> import numpy as np
        >>> X = np.random.rand(3, 2)
        >>> y = np.random.rand(3, 2)
        >>> dataset = TabularDataset(X, y=y)
        >>> dataset[0]
        (tensor([0.7778, 0.3400]), tensor(0.4730))
        ```
        """
        
        super().__init__()

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()

        if self.X.ndim != 2:
            raise ValueError("X data must be a 2D array.")
        
        if self.y.ndim != 1:
            raise ValueError("X data must be a 1D array.")
        
        if self.X.size(0) != self.y.size(0):
            raise ValueError("X and y must of have the same number of rows")

    def get_num_features(self):
        """
        Returns the number of features in the dataset.

        Returns:
            int: Number of features.
        """
        return self.X.size(1)
    
    def __getitem__(self, idx):
        """
        Retrieves the feature and target data for a given index.

        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            tuple: A tuple containing the feature and target data for the given index.
        """
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.X.size(0)
