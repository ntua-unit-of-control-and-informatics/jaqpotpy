from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Iterable, Any, Union


class DOA(ABC):
    """Abstract class for DOA methods"""

    @property
    def __name__(self):
        return NotImplementedError

    @property
    def doa_new(self):
        return self._doa

    @doa_new.setter
    def doa_new(self, value):
        self._doa = value

    @property
    def in_doa(self):
        return self._in_doa

    @in_doa.setter
    def in_doa(self, value):
        self._in_doa = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @abstractmethod
    def fit(self, X: np.array):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: Iterable[Any]) -> Iterable[Any]:
        raise NotImplementedError

    @abstractmethod
    def get_attributes(self):
        raise NotImplementedError


class Leverage(DOA):
    """Implements DOA method leverage.
    Initialized upon training data and holds the doa matrix and the threshold 'A' value.
    Calculates the DOA for a new instance of data or array of data.
    """

    _doa = []
    _in_doa = []

    @property
    def __name__(self):
        return "LeverageDoa"

    def __init__(self) -> None:
        # self._scaler: BaseEstimator = scaler
        self._data: Union[np.array, pd.DataFrame] = None
        self._doa_matrix = None
        self._h_star = None
        self._doa_attributes = None

    def __getitem__(self, key):
        return key

    @property
    def doa_matrix(self):
        return self._doa_matrix

    @doa_matrix.setter
    def doa_matrix(self, value):
        self._doa_matrix = value

    @property
    def h_star(self):
        return self._h_star

    @h_star.setter
    def h_star(self, value):
        self._h_star = value

    def calculate_threshold(self):
        shape = self._data.shape
        h_star = (3 * (shape[1] + 1)) / shape[0]
        self._h_star = h_star

    def calculate_matrix(self):
        x_T = self._data.transpose()
        x_out = x_T.dot(self._data)
        self._doa_matrix = np.linalg.pinv(x_out)

    def fit(self, X: Union[np.array, pd.DataFrame]):
        self._data = self._validate_input(X)
        self.calculate_matrix()
        self.calculate_threshold()
        self._doa_attributes = self.get_attributes()

    def predict(self, new_data: Union[np.array, pd.DataFrame]) -> Iterable[Any]:
        new_data = self._validate_input(new_data)
        doaAll = []
        self._doa = []
        self._in_doa = []
        for nd in new_data:
            d1 = np.dot(nd, self.doa_matrix)
            ndt = np.transpose(nd)
            d2 = np.dot(d1, ndt)
            if d2 < self._h_star:
                in_ad = True
            else:
                in_ad = False
            self._doa.append(d2)
            self._in_doa.append(in_ad)
            doa = {"DOA": d2, "A": self._h_star, "in_doa": in_ad}
            doaAll.append(doa)
        return doaAll

    def _validate_input(self, data: Union[np.array, pd.DataFrame]):
        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
        else:
            return data

    def get_attributes(self):
        return {"doa_matrix": self.doa_matrix, "h_star": self.h_star}


class MeanVar(DOA):
    """Implements Mean and Variance domain of applicability .
    Initialized upon training data and holds the doa mean and the variance of the data.
    Calculates the mean and variance for a new instance of data or array of data and decides if in AD.
    """

    _doa = []
    _in_doa = []

    @property
    def __name__(self):
        return "MeanVar"

    def __init__(self) -> None:
        self._data: np.array = None
        self._bounds = None
        self._doa_attributes = None

    def fit(self, X: np.array):
        self._data = X
        list_m_var = []
        for i in range(self._data.shape[1]):
            list_m_var.append(
                [
                    np.mean(self._data[:, i]),
                    np.std(self._data[:, i]),
                    np.var(self._data[:, i]),
                ]
            )
        self._bounds = np.array(list_m_var)
        self._doa_attributes = self.get_attributes()

    def predict(self, new_data: np.array) -> Iterable[Any]:
        doaAll = []
        self._doa = []
        self._in_doa = []
        in_doa = True
        for nd in new_data:
            for index, row in enumerate(nd):
                bounds = self._bounds[index]
                bounds_data = [bounds[0] - 4 * bounds[1], bounds[0] + 4 * bounds[1]]
                if row >= bounds_data[0] and row <= bounds_data[1]:
                    continue
                else:
                    in_doa = False
            doa = {"in_doa": in_doa}
            doaAll.append(doa)
            self._doa.append(new_data)
            self._in_doa.append(in_doa)
        return doaAll

    def get_attributes(self):
        return {"mean_var": self._bounds}


class BoundingBox(DOA):
    _doa = []
    _in_doa = []

    @property
    def __name__(self):
        return "BoundingBox"

    def __init__(self) -> None:
        self._data: np.array = None
        self._bounding_box = None
        self._doa_attributes = None

    def fit(self, X: np.array):
        self._data = X
        list_m_var = []
        for i in range(self._data.shape[1]):
            list_m_var.append([np.min(self._data[:, i]), np.max(self._data[:, i])])
        self._bounding_box = np.array(list_m_var)
        self._doa_attributes = self.get_attributes()

    def predict(self, new_data: np.array) -> Iterable[Any]:
        doaAll = []
        self._doa = []
        self._in_doa = []
        in_doa = True
        for nd in new_data:
            for index, row in enumerate(nd):
                bounds = self._bounding_box[index]
                bounds_data = [bounds[0], bounds[1]]
                if row >= bounds_data[0] and row <= bounds_data[1]:
                    continue
                else:
                    in_doa = False
            doa = {"in_doa": in_doa}
            doaAll.append(doa)
            self._doa.append(new_data)
            self._in_doa.append(in_doa)
        return doaAll

    def get_attributes(self):
        return {"bounding_box": self._bounding_box}
