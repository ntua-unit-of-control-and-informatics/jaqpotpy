"""
Dataset classes for molecular modelling
"""

from typing import Iterable, Any, Optional
import pandas as pd
from torch.utils.data import Dataset
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from jaqpotpy.descriptors.molecular import MolGraphConvFeaturizer
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
                 task:str = None
                 ) -> None:

        if not(isinstance(smiles_cols, str) or 
              (isinstance(smiles_cols, list) and all(isinstance(item, str) for item in smiles_cols)) or
              (isinstance(smiles_cols, list) and len(smiles_cols) == 0) or
              (smiles_cols is None)):
               raise TypeError("smiles_cols should be either a string, an empty list"
                               "a list of strings, or None") 

        if (smiles_cols is not None) and (featurizer is None):
            raise TypeError("Cannot estimate SMILES descriptors without a featurizer"
                            "Please provide a featurizer") 
        
        #Find the length of each provided column name vector and put everything in lists
        if isinstance(smiles_cols, str):
            self.smiles_cols = [smiles_cols]
            self.smiles_cols_len = 1
        elif isinstance(smiles_cols, list) :
            self.smiles_cols = smiles_cols
            self.smiles_cols_len = len(smiles_cols)
        elif smiles_cols is None:
            self.smiles_cols= [] 
            self.smiles_cols_len = 0
        
        if (featurizer is not None) and not (isinstance(featurizer, MolecularFeaturizer)):
            raise TypeError("featurizer should be a MolecularFeaturizer instance ") 

        super().__init__(df=df, path=path, y_cols=y_cols, x_cols=x_cols, task = task)

        self._validate_column_overlap(self.smiles_cols, self.x_cols, self.y_cols)
        self._validate_column_names(self.smiles_cols, "smiles_cols")
        self._validate_column_names(self.x_cols, "x_cols")
        self._validate_column_names(self.y_cols, "y_cols")
        
        self.init_df = self._df
        self.featurizer = featurizer
        self._featurizer_name = None
        self.smiles = None
        self._x_cols_all = None
        self.create()

    @property
    def featurizer_name(self) -> Iterable[Any]:
        return self.featurizer.__name__

    @featurizer_name.setter
    def featurizer_name(self, value):
        self._featurizer_name = value

    @property
    def x_cols_all(self) -> Iterable[str]:
        return self._x_cols_all

    @x_cols_all.setter
    def x_cols_all(self, value):
        self._x_cols_all = value

    def _validate_column_names(self, cols, col_type):
        """
        Validate if the columns specified in cols are present in the DataFrame.
        
        """
        if len(cols) == 0:
            return
        
        missing_cols = [col for col in cols if col not in self._df.columns]

        if missing_cols:
            raise ValueError(f"The following columns in {col_type} are not present in the DataFrame: {missing_cols}")


    def _validate_column_overlap(self, smiles_cols, x_cols, y_cols):
        smiles_set = set(smiles_cols) if smiles_cols else set()
        x_set = set(x_cols) if x_cols else set()
        y_set = set(y_cols) if y_cols else set()

        overlap_smiles_x = smiles_set & x_set
        overlap_smiles_y = smiles_set & y_set
        overlap_x_y = x_set & y_set

        if len(overlap_smiles_x)>0:
            raise ValueError(f"Overlap found between smiles_cols and x_cols: {overlap_smiles_x}")
        if len(overlap_smiles_y)>0:
            raise ValueError(f"Overlap found between smiles_cols and y_cols: {overlap_smiles_y}")
        if len(overlap_x_y)>0:
            raise ValueError(f"Overlap found between x_cols and y_cols: {overlap_x_y}")


    def create(self):

        if  len(self.smiles_cols) == 1:
            # The method featurize_dataframe needs self.smiles to be pd.Series
            self.smiles = self._df[self.smiles_cols[0]]
            descriptors = self.featurizer.featurize_dataframe(self.smiles)
        elif len(self.smiles_cols)>1:
            featurized_dfs = [self.featurizer.featurize_dataframe(self._df[[col]]) for col in self.smiles_cols]
            descriptors = pd.concat(featurized_dfs, axis=1)
        else:
            #Case where no smiles were provided
            self.smiles = []
            descriptors = []
        
        if len(self.x_cols) == 0:
            if len(descriptors)>0:
                self._x = descriptors
                self.x_cols_all = self._x.columns.tolist()
            else:
                raise ValueError("The design matrix X is empty. Please provide either"
                                  "smiles or other descriptors")

        else:
            self._x =  pd.concat([self._df[self.x_cols] ,  pd.DataFrame(descriptors)] , axis=1)
            self.x_cols_all = self._x.columns.tolist()

        if not self.y_cols:
            self._y = None
        else:
            self._y = self._df[self.y_cols]

        self._df = pd.concat([self._x, self._y], axis = 1)
    
    def copy(self):
        """
        Create a copy of the dataset, including a deep copy of the underlying DataFrame
        and all relevant attributes.
        """
        copied_instance = JaqpotpyDataset(
            df= self.init_df,
            path=self.path,
            y_cols=self.y_cols,
            x_cols=self.x_cols,
            smiles_cols=self.smiles_cols,
            featurizer=self.featurizer,
            task=self.task
        )
        return copied_instance

    def __get_X__(self):
        return self._x.copy()
    
    def __get_Y__(self):
        return self._y.copy()
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self._df]

    def __getitem__(self, idx):
        selected_x = self._df[self._x].iloc[idx].values
        selected_y = self._df[self._y].iloc[idx].to_numpy()
        return selected_x, selected_y

    def __len__(self):
        return self.df.shape[0]

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
        #self.indices: [] = None
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
                    raise TypeError("Please set task (classification / regression).")
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
                    raise TypeError("Please set task (classification / regression).")
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

