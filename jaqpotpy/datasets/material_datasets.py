from jaqpotpy.datasets.dataset_base import MaterialDataset
try:
    from pymatgen.core.composition import Composition
    from pymatgen.core.structure import Lattice, Structure
    from pymatgen.util.typing import CompositionLike
except ModuleNotFoundError:
    Lattice = Structure = CompositionLike = Composition = None

    pass
from jaqpotpy.descriptors.base_classes import MaterialFeaturizer
# from jaqpotpy.descriptors import MolGraphConvFeaturizer, TorchMolGraphConvFeaturizer
from typing import Iterable, Any, Union, Dict
import pandas as pd
import os
import inspect
import numpy as np
import pickle
from torch.utils.data import Dataset


class CompositionDataset(MaterialDataset):
    """
    Reads CSV with compositions, experimental features and endpoints.

    Note
    ----
    Compositions may also be passed as an Iterable of stings representations, if no CSV file is available.
    In this case the endpoints should be passed separetely.
    """
    def __init__(self,  path: str = None, compositions: Union[str, Iterable[CompositionLike]] = None, y_cols: Union[str, Iterable[str]] = None,
                 keep_cols: Union[str, Iterable[str]]=None, y: Iterable[Any] = None, featurizer: MaterialFeaturizer = None, task: str = "regression") -> None:
        """
        A Composition dataset is a Tabular dataset and is defined either by an Iterable of compositions
        or by a CSV. In addition, the endpoints of the dataset should be defined.

        Parameters
        ----------
        path: str
            A path to a CSV file containing the compositions and the endpoints. The compositions in the
            CSV should be in a string format (e.g. "FeO", "Fe3O2", etc.). The file will be passed as a pandas
            DataFrame and descriptors will be the featurization of the compositions in the CSV file.
            If a path is passed, the, the compositions parameter should be filled with the column name
            of the compositions and the y_cols parameter should be filled with the column name of the
            endpoints. In addition, the y parameter may be omitted. Finally, please note that the keep_cols
            parameter can be used to keep and use some other columns from the dataset as well as the
            created descriptors.

        compositions: str or an iterable of pymatgen.util.typing.CompositionLike
            If the path parameter was passed, the compositions parameter should contain the column name of
            the compositions. If the path parameter was not passed then the compositions parameter is mandatory
            and should be an Iterable of pymatgen.util.typing.CompositionLike from which the featurizer
            shall create the descriptors.

        y_cols: str or an iterable of str
            This parameter is taken into account in the case of passing a CSV file. Use the y_cols to
            define the columns in the CSV that contain the endpoints for the modeling. Please note
            that defining the enpoints from the CSV file and not externally isn't mandatory,
            although it is highly recommended.

        keep_cols: str or an iterable of str
            This parameter is taken into account in the case of passing a CSV file. Use the y_cols to
            define the columns in the CSV that should be kept along with the other features that will
            be created from the featurizer.

        y: Iterable
            The y parameter is mandatory in the case where a CSV file was NOT provided. With the y
            parameter the modellers should pass the endpoint data. If a CSV file was passed,
            it is highly recommended to use the y_cols parameter to define the endpoints.

        featurizer: jaqpotpy.descriptors.MaterialFeaturizer
            The featurizer that will take the compositions and create the features for the modelling.

        task: str
            The task of the dataset. Either "regression" (default) or "classification".
        """
        super().__init__(path=path, x_cols=keep_cols, y_cols=y_cols)
        self._y = y
        self._x = None
        self._df = None
        self._x_cols_all = None
        if path:
            self.materials_col = compositions
            self.materials = None
        else:
            self.materials_col = None
            self.materials = compositions
        self._task = task
        self.featurizer: MaterialFeaturizer = featurizer
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

        if self.path:
            data = pd.read_csv(self.path)
            self.materials = data[self.materials_col].to_list()
            y = data[self.y_cols]
            if self.x_cols:
                xs = data[self.x_cols].copy()
            else:
                xs = data.drop(self.y_cols, axis=1).copy()
        else:
            xs = pd.DataFrame()

        if isinstance(self.materials[0], str):
            self._materials_strings = self.materials
        else:
            try:
                self._materials_strings = [Composition(item).formula for item in self.materials]
            except:
                self._materials_strings = []
                raise Warning('Found bad format in a Composition')

        self.indices = list(range(len(self._materials_strings)))
        descriptors = self.featurizer.featurize_dataframe(self.materials)
        if self._y:
            if self.y_cols:
                raise ValueError('Both y and y_cols parameters were passed. Please provide only \
                one of them. The y_cols is strongly recommended')
            else:
                self.y_cols = 'Endpoint'
                y = pd.DataFrame(data=self._y, columns=[self.y_cols])
        else:
            self.y_cols = list(y)
            self.y = list(y)

        if self.x_cols:
            self.df = pd.concat([descriptors, xs, y], axis=1)
            self._x_cols_all = list(descriptors) + list(xs)
        else:
            self.df = pd.concat([descriptors, y], axis=1)
            self._x_cols_all = list(descriptors)

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
        # return self.df[self.y].to_numpy()
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


class StructureDataset(MaterialDataset):
    """
    Creates a dataset for given structures. Structures may be passed as file paths for
    material files (e.g. cif, extxyz etc.) or in terms of any type capable of creating
    a pymatgen.core.Structure object.

    Note
    ----
    Compositions may also be passed as an Iterable of stings representations, if no CSV file is available.
    In this case the endpoints should be passed separetely.
    """
    def __init__(self,  path: Iterable[str] = None, structures: Union[Iterable[Structure], Iterable[Dict]] = None, ext_dataset: Union[pd.DataFrame, str] = None,
                 y_cols: Union[str, Iterable[str]] = None, keep_cols: Union[str, Iterable[str]]=None, y: Iterable[Any] = None,
                 featurizer: MaterialFeaturizer = None, task: str = "regression") -> None:
        """
        A Structure dataset is defined either by an Iterable of material files (containing information on the structure)
        or by an Interable of Structures. In addition, the endpoints of the dataset should be defined.

        Parameters
        ----------
        path: Iterable of str
            An Iterable of material file paths containing information on the structures of the materials
            from which the pymatgen.util.structure.Structure objects will be created. The featurizer shall
            create the descriptors from the structures. If a path is passed, the structures parameter should
            be omitted.

        structures: Iterable of pymatgen.util.structure.Structure or Iterable of dictionaries.
            If the path parameter was passed, the structures parameter should be omitted.
            If the path parameter was not passed then the structures parameter is mandatory
            and should be either an Iterable of pymatgen.util.structure.Structure or an Iterable of dict
            from which pymatgen.util.structure.Structure objects will be created.
            The featurizer shall create the descriptors from the structures.

        y_cols: str or an iterable of str
            This parameter is taken into account in the case of passing a CSV file. Use the y_cols to
            define the columns in the CSV that contain the endpoints for the modeling. Please note
            that defining the enpoints from the CSV file and not externally isn't mandatory,
            although it is highly recommended.

        keep_cols: str or an iterable of str
            This parameter is taken into account in the case of passing a CSV file. Use the y_cols to
            define the columns in the CSV that should be kept along with the other features that will
            be created from the featurizer.

        y: Iterable
            The y parameter is mandatory in the case where a CSV file was NOT provided. With the y
            parameter the modellers should pass the endpoint data. If a CSV file was passed,
            it is highly recommended to use the y_cols parameter to define the endpoints.

        featurizer: jaqpotpy.descriptors.MaterialFeaturizer
            The featurizer that will take the compositions and create the features for the modelling.

        task: str
            The task of the dataset. Either "regression" (default) or "classification".
        """
        super().__init__(path=path, x_cols=keep_cols, y_cols=y_cols)
        self._y = y
        self._x = None
        self._df = None
        self._external = ext_dataset
        self._x_cols_all = None
        self.materials_col = None
        self.materials = structures
        self._task = task
        self.featurizer: MaterialFeaturizer = featurizer
        self.indices: [] = None

    def save(self):
        if self._dataset_name:
            with open(self._dataset_name + ".jdb", 'wb') as f:
                pickle.dump(self, f)
        else:
            with open("jaqpotpy_dataset" + ".jdb", 'wb') as f:
                pickle.dump(self, f)

    def create(self):

        xs = pd.DataFrame()

        if self._external:
            if isinstance(self._external, str):
                data = pd.read_csv(self._external)
            else:
                data = self._external.copy()

            if self.x_cols:
                xs = data[self.x_cols].copy()
            else:
                xs = data.drop(self.y_cols, axis=1).copy()

        if self.path:
            structures = [Structure.from_file(item) for item in self.path]
        else:
            if isinstance(self.materials[0], dict):
                try:
                    structures = [Structure.from_dict(item) for item in self.materials]
                except:
                    try:
                        structures = [Structure(item['lattice'], item['species'], item['coords']) for item in self.materials]
                    except:
                        raise ValueError("The sturctures parameter should be either a dictinary view of a pymatgen.core.Structure or a dictionary \
                        with the keys 'lattice', 'species' and 'coords' and the appropriate values.")
            else:
                structures = self.materials

        self.materials = structures
        self._materials_strings = [item.composition.reduced_composition.formula for item in structures]
        self.indices = list(range(len(self._materials_strings)))
        descriptors = self.featurizer.featurize_dataframe(self.materials)

        if not self.y_cols:
            self.y_cols = 'Endpoint'

        y = pd.DataFrame(data=self._y, columns=[self.y_cols])

        self.y = list(y)

        if self.x_cols:
            self.df = pd.concat([descriptors, xs, y], axis=1)
            self._x_cols_all = list(descriptors) + list(xs)
        else:
            self.df = pd.concat([descriptors, y], axis=1)
            self._x_cols_all = list(descriptors)

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