import os
from typing import Iterable, Any, Optional
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from jaqpotpy.descriptors.molecular import MolGraphConvFeaturizer, TorchMolGraphConvFeaturizer
from jaqpotpy.datasets.dataset_base import BaseDataset


class JaqpotpyDataset(BaseDataset):

    """
    A dataset class for general and molecular data, inheriting from BaseDataset. This class is 
    designed to handle datasets that include molecular structures represented by 
    SMILES strings and use molecular featurizers to generate features from these 
    structures.

    Attributes:
        smiles_col (Optional[str]): The column containing SMILES strings.
        smiles (pd.Series): The SMILES strings extracted from the DataFrame.
        featurizer (Optional[MolecularFeaturizer]): The featurizer used to 
                                                    generate molecular features.
        _featurizer_name (Optional[str]): The name of the featurizer.
        x_colls_all (Optional[Iterable[str]]): All feature columns after featurization.
    """

    def __init__(self, df: pd.DataFrame = None, path: Optional[str] = None,
                 y_cols: Iterable[str] = None,
                 x_cols: Optional[Iterable[str]] = None,
                 smiles_cols: Optional[Iterable[str]] = None,
                 featurizer: Optional[MolecularFeaturizer] = None,
                 task:str = None) -> None:
       
        super().__init__(df=df, path=path, y_cols=y_cols, x_cols=x_cols)
        self.smiles_cols = smiles_cols
        self.smiles = df[self.smiles_cols]
        self.featurizer = featurizer
        self._featurizer_name = None
        self._task = task 

        self._y = df[self.y_cols]
        if x_cols is None:
            # Estimate x_cols by excluding y_cols and smiles_col
            self._x = self.df.drop(columns=y_cols + [smiles_cols])
            self.x_cols = self._x.columns.tolist()
        else:
            self._x = self.df[self.x_cols]
        self._x_cols_all = None
        self.indices: [] = None
        self.create()

    @property
    def featurizer_name(self) -> Iterable[Any]:
        """
        Gets the name of the featurizer.

        Returns:
            Iterable[Any]: The name of the featurizer.
        """
        return self.featurizer.__name__

    @featurizer_name.setter
    def featurizer_name(self, value):
        """
        Sets the name of the featurizer.

        Args:
            value (str): The name of the featurizer.
        """
        self._featurizer_name = value

    @property
    def x_colls_all(self) -> Iterable[str]:
        """
        Gets all feature columns after featurization.

        Returns:
            Iterable[str]: All feature columns after featurization.
        """
        return self._x_cols_all

    @x_colls_all.setter
    def x_colls_all(self, value):
        """
        Sets all feature columns after featurization.

        Args:
            value (Iterable[str]): All feature columns after featurization.
        """
        self._x_cols_all = value

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

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self.df]

    def __getitem__(self, idx):
        # print(self.df[self.X].iloc[idx].values)
        # print(type(self.df[self.X].iloc[idx].values))
        selected_x = self.df[self._x].iloc[idx].values
        selected_y = self.df[self._y].iloc[idx].to_numpy()
        return selected_x, selected_y
    
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
        return len(self.df)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}"
                f"(smiles={True if self.smiles_cols is not None else False}, "
                f"featurizer={self.featurizer_name})")


class TorchGraphDataset(BaseDataset,Dataset):
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
