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
    Reads CSV with SMILES, experimental features, and endpoints.
    Handles datasets with optional SMILES for feature generation.
    """
    def __init__(self, path, feature_columns=None, target_columns=None, smiles_col=None,
                 smiles: Iterable[str] = None, targets: Iterable[Any] = None,
                 features: Iterable[Any] = None, featurizer: MolecularFeaturizer = None,
                 task: str = "regression"):
        super().__init__(path=path, x_cols=feature_columns, y_cols=target_columns)
        self.feature_values = features
        self.target_values = targets
        self.df = None
        self.external_features = None
        self.feature_columns = None
        self.smiles_col = smiles_col
        self.smiles = smiles
        self.smiles_strings = None
        self.targets = targets
        self.task = task
        self.featurizer = featurizer
        self.indices = []

    def save(self):
        dataset_name = self._dataset_name if self._dataset_name else "jaqpotpy_dataset"
        with open(f"{dataset_name}.jdb", 'wb') as f:
            pickle.dump(self, f)

    def create(self):
        print('Hello')
        _, extension = os.path.splitext(self.path)
        if extension == '.csv':
            data = pd.read_csv(self.path)
            if self.smiles_col and self.featurizer:
                smiles = data[self.smiles_col].to_list()
                descriptors = self.featurizer.featurize(smiles)
                if self.feature_columns
                    features = data[self.feature_columns]
                    targets = data[self.target_columns]
                    self.df = pd.concat([descriptors,features, targets], axis=1)
                    self.all_feature_columns = list(descriptors.columns) + self.feature_columns
            else:
                self.df = pd.DataFrame(data[self.target_columns])
            self.feature_values = self.df[self.all_feature_columns].values
            self.target_values = self.df[self.target_columns].values
        else:
            raise ValueError("Only .csv files are supported.")
        return self

    def __get_X__(self):
        return self.df[self.feature_columns].to_numpy()

    def __get_Y__(self):
        return self.df[self.target_columns].to_numpy()

    def __get__(self):
        return self.df

    def __getitem__(self, idx):
        X = self.df[self.feature_columns].iloc[idx].values
        y = self.df[self.target_columns].iloc[idx].to_numpy()
        return X, y

    def __len__(self):
        if self.df is None:
            self.create()
        return len(self.df)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} - features: {self.feature_columns}, targets: {self.target_columns}"


class TorchGraphDataset(MolecularDataset, Dataset):
    """
    Init with smiles and y array
    """
    def __init__(self, smiles=Iterable[str]
                 , y: Iterable[Any] = Iterable[Any], featurizer: MolecularFeaturizer = MolGraphConvFeaturizer(), task=str, streaming: bool = False) -> None:
        super(TorchGraphDataset, self).__init__()
        self._y = y
        self._dataset_name = None
        self._df = []
        self._x_cols_all = None
        self.smiles = smiles
        self._smiles_strings = None
        self.ys = y
        self._task = task
        self.featurizer: MolecularFeaturizer = featurizer
        self.indices: [] = None
        self.streaming = streaming
        # self.create()

    def create(self):
        if self.streaming is False:
            import torch
            from torch_geometric.data import Data
            descriptors = self.featurizer.featurize(datapoints = self.smiles)
            for i, g in enumerate(descriptors):
                if self._task == 'regression':
                    dato = Data(x=torch.FloatTensor(g.node_features)
                                , edge_index=torch.LongTensor(g.edge_index)
                                , edge_attr= torch.LongTensor(g.edge_features) if g.edge_features is not None else g.edge_features
                                , num_nodes=g.num_nodes, y=torch.Tensor([self.y[i]]))
                    self._df.append(dato)
                elif self._task == 'classification':
                    dato = Data(x=torch.FloatTensor(g.node_features)
                                , edge_index=torch.LongTensor(g.edge_index)
                                , edge_attr=torch.LongTensor(g.edge_features) if g.edge_features is not None else g.edge_features
                                , num_nodes=g.num_nodes, y=torch.LongTensor([self.y[i]]))
                    self._df.append(dato)
                else:
                    raise Exception("Please set task (classification / regression).")
            self._smiles_strings = self.smiles
            self.X = ['TorchMolGraph']
            self.y = ['Y']
            return self
        else:
            ys = pd.DataFrame(self.ys, columns=['Y'])
            self.df = pd.concat([self.smiles, ys], axis=1)
            return self

    def __get__(self):
        return self.df

    def __getitem__(self, idx):
        if self.streaming is False:
            return self.df[idx]
        else:
            tempdf = []
            import torch
            from torch_geometric.data import Data
            smile = self.smiles[idx]
            descriptors = self.featurizer.featurize(smile)
            for i, g in enumerate(descriptors):
                if self._task == 'regression':
                    dato = Data(x=torch.FloatTensor(g.node_features)
                                , edge_index=torch.LongTensor(g.edge_index)
                                , edge_attr= torch.LongTensor(g.edge_features) if g.edge_features is not None else g.edge_features
                                , num_nodes=g.num_nodes, y=torch.Tensor([self.y[i]]))
                    tempdf.append(dato)
                elif self._task == 'classification':
                    dato = Data(x=torch.FloatTensor(g.node_features)
                                , edge_index=torch.LongTensor(g.edge_index)
                                , edge_attr=torch.LongTensor(g.edge_features) if g.edge_features is not None else g.edge_features
                                , num_nodes=g.num_nodes, y=torch.LongTensor([self.y[i]]))
                    tempdf.append(dato)
                else:
                    raise Exception("Please set task (classification / regression).")
            self._smiles_strings = self.smiles
            self.X = ['TorchMolGraph']
            self.y = ['Y']
            return tempdf[idx]

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
                 featurizer: MolecularFeaturizer = None, task: str = 'regression', streaming: bool = False) -> None:
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
        self.streaming = streaming

        # self.create()

    def create(self):
        if self.streaming is False:
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
        else:
            ys = pd.DataFrame(self.ys, columns=['Y'])
            self.df = pd.concat([self.smiles, ys], axis=1)
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
        if self.streaming is False:
            if self._task != "generation":
                if self.X == ['Sequence']:
                    X = self.df[self.X].iloc[idx].values[0]
                elif self.X == ['OneHotSequence']:
                    X = self.df[self.X].iloc[idx].values[0]
                elif self.X == ['SmilesImage']:
                    X = self.df[self.X].iloc[idx].values[0].transpose(2, 0, 1)
                else:
                    X = self.df[self.X].iloc[idx].values
                y = self.df[self.y].iloc[idx].to_numpy()
                return X, y
            else:
                try:
                    node_features = self.df[self.X].iloc[idx].values[0].node_features
                    adjacency_matrix = self.df[self.X].iloc[idx].values[0].adjacency_matrix
                    import torch
                    X = torch.Tensor(node_features), torch.Tensor(adjacency_matrix)
                except Exception as e:
                    print(idx)
                    print(self.df[self.X].iloc[idx])
                    print(str(e))
                return X, idx, (torch.Tensor(node_features), torch.Tensor(adjacency_matrix))
        elif self.streaming is True:
            smiles = self.smiles[idx]
            descriptors = self.featurizer.featurize_dataframe(smiles)
            if self._task != "generation":
                y = self.ys[idx]
                ys = pd.DataFrame([y], columns=['Y'])
                temp_df = pd.concat([descriptors, ys], axis=1)
                self._smiles_strings = self.smiles
                self.X = list(descriptors)
                self.y = ['Y']
            else:
                temp_df = pd.concat([descriptors], axis=1)
                self.X = list(descriptors)
            if self._task != "generation":
                if self.X == ['Sequence']:
                    X = temp_df[self.X].iloc[0].values[0]
                elif self.X == ['OneHotSequence']:
                    X = temp_df[self.X].iloc[0].values[0]
                elif self.X == ['SmilesImage']:
                    X = temp_df[self.X].iloc[0].values[0].transpose(2, 0, 1)
                else:
                    X = temp_df[self.X].iloc[0].values
                y = temp_df[self.y].iloc[0].to_numpy()
                return X, y
            else:
                try:
                    node_features = temp_df[self.X].iloc[0].values[0].node_features
                    adjacency_matrix = temp_df[self.X].iloc[0].values[0].adjacency_matrix
                    import torch
                    X = torch.Tensor(node_features), torch.Tensor(adjacency_matrix)
                except Exception as e:
                    print(idx)
                    print(temp_df[self.X].iloc[0])
                    print(str(e))
                return X, idx, (torch.Tensor(node_features), torch.Tensor(adjacency_matrix))


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
