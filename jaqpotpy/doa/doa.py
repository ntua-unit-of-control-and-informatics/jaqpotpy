from abc import ABC
import pandas as pd
import numpy as np
from typing import Iterable, Any
import math

def calculate_a(X):
    shape = X.shape
    a = (3 * (shape[1] + 1)) / shape[0]
    return a


def calculate_doa_matrix(X):
    x_T = X.transpose()
    x_out = x_T.dot(X)
    x_out_inv = pd.DataFrame(np.linalg.pinv(x_out.values), x_out.columns, x_out.index)
    return x_out_inv


def calc_doa(doa_matrix, new_data):
    doaAll = []
    for nd in new_data:
        d1 = np.dot(nd, doa_matrix)
        ndt = np.transpose(nd)
        d2 = np.dot(d1, ndt)
        doa = {'DOA': d2}
        doaAll.append(doa)
    return doaAll


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

    def predict(self, data):
        raise NotImplementedError


class Leverage(DOA, ABC):
    """
    Implements DOA method leverage.
    Initialized upon training data and holds the doa matrix and the threshold 'A' value.
    Calculates the DOA for a new instance of data or array of data.
    """
    def __init__(self) -> None:
        # self._scaler: BaseEstimator = scaler
        self._data: np.array = None
        self._doa_matrix = None
        self._a = None

    def __getitem__(self):
        return self

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

    def fit(self, X: np.array):
        # self._scaler.fit(X)
        # self._data = self._scaler.transform(X)
        self._data = X
        self.calculate_matrix()
        self.calculate_threshold()

    def calculate(self, new_data: np.array) -> Iterable[Any]:
        doaAll = []
        # new_data = self._scaler.transform(new_data)
        for nd in new_data:
            d1 = np.dot(nd, self.doa_matrix)
            ndt = np.transpose(nd)
            d2 = np.dot(d1, ndt)
            if d2 < self._a:
                in_ad = True
            else:
                in_ad = False
            doa = {'DOA': d2, 'A': self._a, 'IN': in_ad}
            doaAll.append(doa)
        return doaAll


class MeanVar(DOA, ABC):
    """
    Implements Mean and Variance domain of applicability .
    Initialized upon training data and holds the doa mean and the variance of the data.
    Calculates the mean and variance for a new instance of data or array of data and decides if in AD.
    """
    def __init__(self) -> None:
        # self._scaler: BaseEstimator = scaler
        self._data: np.array = None
        self._doa_matrix = None
        self._a = None

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