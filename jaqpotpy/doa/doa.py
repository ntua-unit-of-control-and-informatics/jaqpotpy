from abc import ABC
import pandas as pd
import numpy as np
from typing import Iterable, Any
import math
from jaqpotpy.descriptors.molecular import RDKitDescriptors, MordredDescriptors
import pickle
# import dill

# def calculate_a(X):
#     shape = X.shape
#     a = (3 * (shape[1] + 1)) / shape[0]
#     return a


# def calculate_doa_matrix(X):
#     x_T = X.transpose()
#     x_out = x_T.dot(X)
#     x_out_inv = pd.DataFrame(np.linalg.pinv(x_out.values), x_out.columns, x_out.index)
#     return x_out_inv


# def calc_doa(doa_matrix, new_data):
#     doaAll = []
#     for nd in new_data:
#         d1 = np.dot(nd, doa_matrix)
#         ndt = np.transpose(nd)
#         d2 = np.dot(d1, ndt)
#         doa = {'DOA': d2}
#         doaAll.append(doa)
#     return doaAll


class DOA(object):
    """
    Abstract class for DOA methods
    """
    def calculate_threshold(self):
        raise NotImplementedError

    def calculate_matrix(self):
        raise NotImplementedError

    def calculate(self, data: Iterable[Any]) -> Iterable[Any]:
        raise NotImplementedError

    def fit(self, X: np.array):
        raise NotImplementedError

    def predict(self, data: Iterable[Any]) -> Iterable[Any]:
        raise NotImplementedError


class Leverage(DOA, ABC):
    """
    Implements DOA method leverage.
    Initialized upon training data and holds the doa matrix and the threshold 'A' value.
    Calculates the DOA for a new instance of data or array of data.
    """
    _doa = []
    _in = []

    @property
    def __name__(self):
        return 'LeverageDoa'

    def __init__(self) -> None:
        # self._scaler: BaseEstimator = scaler
        self._data: np.array = None
        self._doa_matrix = None
        self._a = None

    def __getitem__(self,key):
        return key

    @property
    def doa_new(self):
        return self._doa

    @doa_new.setter
    def doa_new(self, value):
        self._doa = value

    @property
    def IN(self):
        return self._in

    @IN.setter
    def IN(self, value):
        self._in = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def doa_matrix(self):
        return self._doa_matrix

    @doa_matrix.setter
    def doa_matrix(self, value):
        self._doa_matrix = value

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = value

    def calculate_threshold(self):
        shape = self._data.shape
        a = (3 * (shape[1] + 1)) / shape[0]
        self._a = a

    def calculate_matrix(self):
        x_T = self._data.transpose()
        x_out = x_T.dot(self._data)
        self._doa_matrix = np.linalg.pinv(x_out)

    def fit(self, X: np.array):
        self._data = X
        self.calculate_matrix()
        self.calculate_threshold()

    def predict(self, new_data: np.array) -> Iterable[Any]:
        doaAll = []
        self._doa = []
        self._in = []
        for nd in new_data:
            d1 = np.dot(nd, self.doa_matrix)
            ndt = np.transpose(nd)
            d2 = np.dot(d1, ndt)
            if d2 < self._a:
                in_ad = True
            else:
                in_ad = False
            self._doa.append(d2)
            self._in.append(in_ad)
            doa = {'DOA': d2, 'A': self._a, 'IN': in_ad}
            doaAll.append(doa)
        return doaAll


class MeanVar(DOA, ABC):
    """
    Implements Mean and Variance domain of applicability .
    Initialized upon training data and holds the doa mean and the variance of the data.
    Calculates the mean and variance for a new instance of data or array of data and decides if in AD.
    """
    _doa = []
    _in = []

    @property
    def __name__(self):
        return 'MeanVar'

    def __init__(self) -> None:
        # self._scaler: BaseEstimator = scaler
        self._data: np.array = None
        self._doa_matrix = None
        self._a = None

    @property
    def doa_new(self):
        return self._doa

    @doa_new.setter
    def doa_new(self, value):
        self._doa = value

    @property
    def IN(self):
        return self._in

    @IN.setter
    def IN(self, value):
        self._in = value

    @property
    def doa_matrix(self):
        return self._doa_matrix

    @doa_matrix.setter
    def doa_matrix(self, value):
        self._doa_matrix = value

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def fit(self, X: np.array):
        self._data = X
        # self._scaler.fit(X)
        # self._data = self._scaler.transform(X)
        columns = list(zip(*self._data))
        shape = X.shape
        list_m_var = []
        for i in range(shape[1]):
            list_m_var.append([np.mean(columns[i]), np.std(columns[i]), np.var(columns[i])])
        self._data = np.array(list_m_var)
        self._doa_matrix = np.array(list_m_var)
        self._a = np.array(list_m_var)

    def calculate(self, new_data: np.array) -> Iterable[Any]:
        doaAll = []
        self._doa = []
        self._in = []
        # new_data = self._scaler.transform(new_data)
        in_doa = True
        for nd in new_data:
            for index, row in enumerate(nd):
                bounds = self._data[index]
                bounds_data = [bounds[0]-4*bounds[1], bounds[0]+4*bounds[1]]
                if row >= bounds_data[0] and row <= bounds_data[1]:
                    continue
                else:
                    in_doa = False
            # if len(new_data[0]) > 100 and many > 5:
            #     in_doa = False
            doa = {'IN': in_doa}
            doaAll.append(doa)
        return doaAll

    def predict(self, new_data: np.array) -> Iterable[Any]:
        doaAll = []
        self._doa = []
        self._in = []
        # new_data = self._scaler.transform(new_data)
        in_doa = True
        for nd in new_data:
            for index, row in enumerate(nd):
                bounds = self._data[index]
                bounds_data = [bounds[0]-4*bounds[1], bounds[0]+4*bounds[1]]
                if row >= bounds_data[0] and row <= bounds_data[1]:
                    continue
                else:
                    in_doa = False
            # if len(new_data[0]) > 100 and many > 5:
            #     in_doa = False
            doa = {'IN': in_doa}
            doaAll.append(doa)
            self._doa.append(new_data)
            self._in.append(in_doa)
        return doaAll


class SmilesLeverage(DOA, ABC):
    """
    Implements DOA method leverage given an array of smiles.
    Descriptors and data matrix is calculated with rdkit descriptors
    Initialized upon training data and holds the doa matrix and the threshold 'A' value.
    Calculates the DOA for a new instance of data or array of data.
    """
    _doa = []
    _in = []

    @property
    def __name__(self):
        return 'SmilesLeverage'

    def __init__(self) -> None:
        # self._scaler: BaseEstimator = scaler
        self._smiles = None
        self._data: np.array = None
        self._doa_matrix = None
        self._a = None
        # self.featurizer = MordredDescriptors(ignore_3D=True)
        self.featurizer = RDKitDescriptors(use_fragment=False, ipc_avg=False)

    def __getitem__(self,key):
        return key

    @property
    def doa_new(self):
        return self._doa

    @doa_new.setter
    def doa_new(self, value):
        self._doa = value

    @property
    def IN(self):
        return self._in

    @IN.setter
    def IN(self, value):
        self._in = value

    @property
    def smiles(self):
        return self._smiles

    @smiles.setter
    def smiles(self, value):
        self._smiles = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def doa_matrix(self):
        return self._doa_matrix

    @doa_matrix.setter
    def doa_matrix(self, value):
        self._doa_matrix = value

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = value

    def calculate_threshold(self):
        shape = self._data.shape
        a = (3 * (shape[1] + 1)) / shape[0]
        self._a = a

    def calculate_matrix(self):
        x_T = self._data.transpose()
        x_out = x_T.dot(self._data)
        self._doa_matrix = np.linalg.pinv(x_out)
        # self.doa_matrix = x_out #pd.DataFrame(np.linalg.pinv(x_out.values), x_out.columns, x_out.index)

    def fit(self, smiles: Iterable[str]):
        # self._scaler.fit(X)
        # self._data = self._scaler.transform(X)
        self._smiles = smiles
        from jaqpotpy.descriptors.molecular import RDKitDescriptors
        featurizer = RDKitDescriptors(use_fragment=False, ipc_avg=False)
        self._data = featurizer.featurize(smiles)
        self.calculate_matrix()
        self.calculate_threshold()

    def predict(self, smiles: Iterable[str]) -> Iterable[Any]:
        doaAll = []
        self._doa = []
        self._in = []
        # new_data = self._scaler.transform(new_data)
        from jaqpotpy.descriptors.molecular import RDKitDescriptors
        featurizer = RDKitDescriptors(use_fragment=False, ipc_avg=False)
        new_data = featurizer.featurize(smiles)
        # new_data = self.featurizer.featurize(smiles)
        for nd in new_data:
            d1 = np.dot(nd, self.doa_matrix)
            ndt = np.transpose(nd)
            d2 = np.dot(d1, ndt)
            if d2 < self._a:
                in_ad = True
            else:
                in_ad = False
            self._doa.append(d2)
            self._in.append(in_ad)
            doa = {'DOA': d2, 'A': self._a, 'IN': in_ad}
            doaAll.append(doa)
        return doaAll
