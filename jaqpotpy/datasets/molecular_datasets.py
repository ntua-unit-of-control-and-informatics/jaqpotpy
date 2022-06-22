from jaqpotpy.datasets.dataset_base import MolecularDataset
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
try:
    from jaqpotpy.descriptors.molecular import MolGraphConvFeaturizer, TorchMolGraphConvFeaturizer
except ImportError:
    pass
from typing import Iterable, Any
import pandas as pd
import os
import inspect
import numpy as np
import pickle
from torch.utils.data import Dataset


class MolecularTabularDataset(MolecularDataset):
    """
    Reads CSV with smiles, experimental features and endpoints
    """
    def __init__(self, path, x_cols=None, y_cols=Iterable[Any], smiles_col=None, smiles=Iterable[str]
                 , y: Iterable[Any] = Iterable[Any], X: Iterable[Any] = None,
                 featurizer: MolecularFeaturizer = None, task: str = "regression") -> None:
        # super(SmilesTabularDataset, path, x_cols, y_cols).__init__(path, x_cols, y_cols)
        super(MolecularTabularDataset, self).__init__(path=path, x_cols=x_cols, y_cols=y_cols)
        self._y = y
        self._x = X
        self._df = None
        self._external = None
        self._x_cols_all = None
        self.smiles_col = smiles_col
        self.smiles = smiles
        self._smiles_strings = None
        self.ys = y
        self._task = task
        self.featurizer: MolecularFeaturizer = featurizer
        self.indices: [] = None
        # self.create()

    def save(self):
        if self._dataset_name:
            with open(self._dataset_name + ".jdb", 'wb') as f:
                pickle.dump(self, f)
        else:
            with open("jaqpotpy_dataset" + ".jdb", 'wb') as f:
                pickle.dump(self, f)

    def create(self):
        name, extension = os.path.splitext(self.path)
        if extension == '.csv':
            data = pd.read_csv(self.path)
            smiles = data[self.smiles_col].to_list()
            self._smiles_strings = smiles
            self.smiles = smiles
            descriptors = self.featurizer.featurize_dataframe(smiles)
            # print(len(list(descriptors)))
            if self.x_cols:
                self._external = self.x_cols
                xs = data[self.x_cols]
                y = data[self.y_cols]
                self.df = pd.concat([descriptors, xs, y], axis=1)
                self._x_cols_all = list(descriptors) + list(xs)
                self.y_cols = list(y)
                self.y = list(y)
            else:
                y = data[self.y_cols]
                self.df = pd.concat([descriptors, y], axis=1)
                self._x_cols_all = list(descriptors)
                self.y_cols = list(y)
                self.y = list(y)
            if self.X is None:
                if self.x_cols:
                    xs = data[self.x_cols]
                    if not xs.empty:
                        self._x = list(descriptors) + list(xs)
                    else:
                        self._x = list(descriptors)
                else:
                    self._x = list(descriptors)
        return self

    def __get_X__(self):
        return self.df[self.X].to_numpy()

    def __get_Y__(self):
        return self.df[self.y].to_numpy()

    def __get__(self):
        return self.df

    def __getitem__(self, idx):
        # print(self.df[self.X].iloc[idx].values)
        # print(type(self.df[self.X].iloc[idx].values))
        X = self.df[self.X].iloc[idx].values
        y = self.df[self.y].iloc[idx].to_numpy()
        return X, y

    def __len__(self):
        if self.df is None:
            self.create()
        return len(self.df)

    def __repr__(self) -> str:
        # args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
        # args_names = [arg for arg in args_spec.args if arg != 'self']
        # args_info = ''
        # for arg_name in args_names:
        #   value = self.__dict__[arg_name]
        #   # for str
        #   if isinstance(value, str):
        #     value = "'" + value + "'"
        #   # for list
        return self.__class__.__name__


class TorchGraphDataset(MolecularDataset, Dataset):
    """
    Init with smiles and y array
    """
    def __init__(self, smiles=Iterable[str]
                 , y: Iterable[Any] = Iterable[Any], task=str) -> None:
        super(TorchGraphDataset, self).__init__()
        self._y = y
        self._dataset_name = None
        self._df = []
        self._x_cols_all = None
        self.smiles = smiles
        self._smiles_strings = None
        self.ys = y
        self._task = task
        self.featurizer: MolecularFeaturizer = MolGraphConvFeaturizer(use_edges=True)
        self.indices: [] = None
        # self.create()

    def create(self):
        import torch
        from torch_geometric.data import Data
        descriptors = self.featurizer.featurize(self.smiles)
        for i, g in enumerate(descriptors):
            if self._task == 'regression':
                dato = Data(x=torch.FloatTensor(g.node_features)
                            , edge_index=torch.LongTensor(g.edge_index)
                            , edge_attr=g.edge_features
                            , num_nodes=g.num_nodes, y=torch.Tensor([self.y[i]]))
                self._df.append(dato)
            elif self._task == 'classification':
                dato = Data(x=torch.FloatTensor(g.node_features)
                            , edge_index=torch.LongTensor(g.edge_index)
                            , edge_attr=g.edge_features
                            , num_nodes=g.num_nodes, y=torch.LongTensor([self.y[i]]))
                self._df.append(dato)
            else:
                raise Exception("Please set task (classification / regression).")
        self._smiles_strings = self.smiles
        self.X = ['TorchMolGraph']
        self.y = ['Y']
        return self

    def __get__(self):
        return self.df

    def __getitem__(self, idx):
        return self.df[idx]

    def __len__(self):
        return len(self.df)

    def __repr__(self) -> str:
        # args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
        # args_names = [arg for arg in args_spec.args if arg != 'self']
        # args_info = ''
        # for arg_name in args_names:
        #   value = self.__dict__[arg_name]
        #   # for str
        #   if isinstance(value, str):
        #     value = "'" + value + "'"
          # for list
        return self.__class__.__name__


class SmilesDataset(MolecularDataset):
    """
    Init with smiles and y array
    """
    def __init__(self, x_cols=None, smiles=Iterable[str]
                 , y: Iterable[Any] = Iterable[Any], X: Iterable[Any] = None,
                 featurizer: MolecularFeaturizer = None, task: str = 'regression') -> None:
        super(SmilesDataset, self).__init__(x_cols=x_cols)
        self._y = y
        self._x = X
        self._dataset_name = None
        self._df = None
        self._x_cols_all = None
        self.smiles = smiles
        self._smiles_strings = None
        self._external = None
        self.ys = y
        self.featurizer: MolecularFeaturizer = featurizer
        self.indices: [] = None
        self._task = task

        # self.create()

    def create(self):
        descriptors = self.featurizer.featurize_dataframe(self.smiles)
        if self._task != "generation":
            ys = pd.DataFrame(self.ys, columns=['Y'])
            self.df = pd.concat([descriptors, ys], axis=1)
            self._smiles_strings = self.smiles
            self.X = list(descriptors)
            self.y = ['Y']
        else:
            self.df = pd.concat([descriptors], axis=1)
            self._smiles_strings = self.smiles
            self.X = list(descriptors)
        return self

    def save(self):
        if self._dataset_name:
            with open(self._dataset_name + ".jdb", 'wb') as f:
                pickle.dump(self, f)
        else:
            with open("jaqpotpy_dataset" + ".jdb", 'wb') as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def __get_X__(self):
        return self.df[self.X].to_numpy()

    def __get_Y__(self):
        return self.df[self.y].to_numpy()

    def __get__(self):
        return self.df

    def __getitem__(self, idx):
        if self.X == ['Sequence']:
            X = self.df[self.X].iloc[idx].values[0]
        elif self.X == ['OneHotSequence']:
            # print(self.df[self.X].iloc[idx].values.shape)
            # print(self.df[self.X].iloc[idx].values[0].shape)
            X = self.df[self.X].iloc[idx].values[0]
        elif self.X == ['SmilesImage']:
            X = self.df[self.X].iloc[idx].values[0].transpose(2, 0, 1)
            # print(X.shape)
            # X = self.df[self.X].iloc[idx].values.transpose((2, 0, 1))
        else:
            X = self.df[self.X].iloc[idx].values
        y = self.df[self.y].iloc[idx].to_numpy()
        return X, y

    def __len__(self):
        if self.df is None:
            self.create()
        return len(self.df)

    def __repr__(self) -> str:
        # args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
        # args_names = [arg for arg in args_spec.args if arg != 'self']
        # args_info = ''
        # for arg_name in args_names:
        #   value = self.__dict__[arg_name]
        #   # for str
        #   if isinstance(value, str):
        #     value = "'" + value + "'"
        #   # for list
        return self.__class__.__name__
