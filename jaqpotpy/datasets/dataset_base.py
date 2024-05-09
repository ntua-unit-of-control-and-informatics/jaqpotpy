"""
Dataset base classes
"""
from typing import Any
import inspect
from typing import Iterable
import pickle
from typing import List, Optional
import pandas as pd

class BaseDataset(object):
    """
    Astract class for datasets
    """
    def __init__(self, path=None, x_cols=None, y_cols=None) -> None:
        self._Y = None
        self._X = None
        self._dataset_name = None
        self._df = None
        self._x_cols_all = None
        self.path = path
        self.x_cols = x_cols
        self.y_cols = y_cols
        self._task = "regression"
        self.featurizer = None
        self._featurizer_name = None
        self._external = None

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        self._task = value

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value):
        self._dataset_name = value

    @property
    def featurizer_name(self) -> Iterable[Any]:
        return self.featurizer.__name__

    @property
    def x_colls_all(self) -> Iterable[str]:
        return self._x_cols_all

    @x_colls_all.setter
    def x_colls_all(self, value):
        self._x_cols_all = value

    @property
    def X(self) -> Iterable[str]:
        return self._x

    @X.setter
    def X(self, value):
        self._x = value

    @property
    def y(self) -> Iterable[str]:
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def external(self) -> Iterable[str]:
        return self._external

    @external.setter
    def external(self, value):
        self._external = value

    @property
    def df(self) -> Any:
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    @featurizer_name.setter
    def featurizer_name(self, value):
        self._featurizer_name = value

    def create(self):
        raise NotImplementedError("Need implementation")

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


class MolecularDataset(BaseDataset):
    def __init__(self, path=None, smiles_col=None, x_cols=None, y_cols=None, smiles=None) -> None:
        self.smiles = smiles
        self._smiles_strings = None
        self.smiles_col = smiles_col
        super().__init__(path, x_cols, y_cols)

    @property
    def smiles_strings(self) -> Iterable[str]:
        return self._smiles_strings

    @smiles_strings.setter
    def smiles_strings(self, value):
        self._smiles_strings = value

    def save(self):
        if self._dataset_name:
            with open(self._dataset_name + ".jdata", 'wb') as f:
                pickle.dump(self, f)
        else:
            with open("jaqpot_dataset" + ".jdata", 'wb') as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


class MaterialDataset(BaseDataset):
    def __init__(self, path=None, materials_col=None, x_cols=None, y_cols=None, materials=None) -> None:
        self.materials = materials
        self._materials_strings = None
        self.materials_col = materials_col
        super().__init__(path, x_cols, y_cols)


    @property
    def materials_strings(self) -> Iterable[str]:
        return self._materials_strings

    @materials_strings.setter
    def materials_strings(self, value):
        self._materials_strings = value

    def save(self):
        if self._dataset_name:
            with open(self._dataset_name + ".jdata", 'wb') as f:
                pickle.dump(self, f)
        else:
            with open("jaqpot_dataset" + ".jdata", 'wb') as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

class ImageDataset(BaseDataset):
    def __init__(self, path=None,  x_cols=None, y_cols=None) -> None:
        super().__init__(path=path, x_cols=x_cols, y_cols=y_cols)

    def save(self):
        if self._dataset_name:
            with open(self._dataset_name + ".jdata", 'wb') as f:
                pickle.dump(self, f)
        else:
            with open("jaqpot_dataset" + ".jdata", 'wb') as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)



    
    
class NumericalVectorDataset(BaseDataset):
    """
    A subclass of BaseDataset for handling datasets composed of numerical vectors.
    """
    def __init__(self, vectors: Optional[List[List[float]]] = None, targets: Optional[List[float]] = None, path: Optional[str] = None):
        super().__init__(path)
        self.vectors = vectors if vectors is not None else []
        self.targets = targets if targets is not None else []
        self.x_cols = [f'feature_{i}' for i in range(len(self.vectors[0]))] if self.vectors else []
        self.y_cols = ['target'] if self.targets else []
        self._task = "regression"  # Default task; can be changed to classification if needed
    
    def save(self):
        if self._dataset_name:
            with open(self._dataset_name + ".jdb", 'wb') as f:
                pickle.dump(self, f)
        else:
            with open("jaqpotpy_dataset" + ".jdb", 'wb') as f:
                pickle.dump(self, f)
        

    def create(self):
        """
        Creates the DataFrame from the input vectors and targets.
        """
        if self.path:
            # Load data from a CSV file if a path is provided
            self.df = pd.read_csv(self.path)
        else:
            # Create DataFrame from provided lists
            self.df = pd.DataFrame(self.vectors, columns=self.x_cols)
            if self.targets:
                self.df['target'] = self.targets
        
        self.x_cols = self.df.columns[:-1]  # All columns except the last one
        self.X = self.df[self.x_cols].to_numpy()  # Storing feature data as numpy array
        self.y_cols = self.df.columns[-1]    # The last column
        #self.y = self.df['target'] if 'target' in self.df.columns else None
        self.y = self.df[self.df.columns[-1]].to_numpy()  # Storing target data as numpy array

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_vectors={len(self.vectors)}, num_features={len(self.x_cols) if self.x_cols else 0})"

    def __get_X__(self):
        return self.X

    def __get_Y__(self):
        return self.y

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

