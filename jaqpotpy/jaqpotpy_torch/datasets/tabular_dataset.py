from torch.utils.data import Dataset
import torch
import pandas as pd


class TabularDataset(Dataset):
    def __init__(self, X, y=None):
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
        return self.X.size(1)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        # return {'X': self.X[idx], 'y': self.y[idx]}
    
    def __len__(self):
        return self.X.size(0)
