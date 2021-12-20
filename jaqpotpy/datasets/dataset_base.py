"""
Dataset base classes
"""
import numpy as np
from typing import Any, Dict, Iterable, Optional, Tuple, Union, cast
from jaqpotpy.descriptors.base_classes import Featurizer
import os
# import pandas as pd
import csv


class SmilesDataset(object):
    """
    Astract class of datasets
    """
    def __init__(self, path, smiles_col=None,  x_cols=None, y_cols=None) -> None:
        self.path = path
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.smiles_col = smiles_col

    @property
    def X(self) -> Iterable[Any]:
        return self.X

    @property
    def Y(self) -> Iterable[Any]:
        return self.Y

    @property
    def ids(self) -> Iterable[Any]:
        return self.ids

    @property
    def x_cols(self) -> Iterable[Any]:
        return self.x_cols

    @property
    def y_cols(self) -> Iterable[Any]:
        return self.y_cols

    @x_cols.setter
    def x_cols(self, value):
        self._x_cols = value

    @y_cols.setter
    def y_cols(self, value):
        self._y_cols = value


class SmilesTabularDataset(SmilesDataset):
    """
    Reads CSV with smiles and endpoints
    """
    def __init__(self, path, x_cols=None, y_cols=None, smiles_col=None, smiles=Iterable[str], y=Iterable[Any],
                 featurizer: Featurizer = None) -> None:
        # super(SmilesTabularDataset, path, x_cols, y_cols).__init__(path, x_cols, y_cols)
        super(SmilesTabularDataset, self).__init__(path, x_cols, y_cols)
        self.y = None
        self.x = None
        self.smiles_col = smiles_col
        self.featurizer: Featurizer = featurizer


    @property
    def featurizer_name(self) -> Iterable[Any]:
        return self.featurizer.__name__

    @property
    def X(self) -> Iterable[Any]:
        return self.x

    @property
    def Y(self) -> Iterable[Any]:
        return self.y

    def create(self) -> None:
        # mydict = {}
        print(self.smiles_col)
        # print(self.x_cols)
        # print(self.smiles_col)
        name, extension = os.path.splitext(self.path)
        if extension == '.csv':
            filename = open(self.path, 'r')
            file = csv.DictReader(filename)
            for col in file:
                print(col)
            # with open(self.path, mode='r') as reader:
            #     d = {}
            #     for row in reader:
            #         k, v = row
            #         d[k] = v
            # print(d)
            # df = pd.read_csv(self.path)
        # return SmilesTabularDataset()



    #     self.path = path
    #     self.x_cols = x_cols
    #     self.y_cols = y_cols
    #
    # @property
    # def x_cols(self) -> Iterable[Any]:
    #     return self.x_cols
    #
    # @property
    # def y_cols(self) -> Iterable[Any]:
    #     return self.y_cols
    #
    # @x_cols.setter
    # def x_cols(self, value):
    #     self._x_cols = value
    #
    # @y_cols.setter
    # def y_cols(self, value):
    #     self._y_cols = value
    @featurizer_name.setter
    def featurizer_name(self, value):
        self._featurizer_name = value