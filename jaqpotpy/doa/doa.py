from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Iterable, Any, Union

from jaqpotpy.api.openapi.models.bounding_box_doa import BoundingBoxDoa
from jaqpotpy.api.openapi.models.leverage_doa import LeverageDoa
from jaqpotpy.api.openapi.models.mean_var_doa import MeanVarDoa


class DOA(ABC):
    """Abstract class for DOA methods"""

    @property
    def __name__(self):
        return NotImplementedError

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

    def _validate_input(self, data: Union[np.array, pd.DataFrame]):
        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
        else:
            return data


class Leverage(DOA):
    """
    Leverage class for Domain of Applicability (DOA) calculation using leverage method.
    Attributes:
        _doa (list): List to store leverage values.
        _in_doa (list): List to store boolean values indicating if data points are within DOA.
        _data (Union[np.array, pd.DataFrame]): Input data used for DOA calculation.
        _doa_matrix (np.array): Matrix used for leverage calculation.
        _h_star (float): Threshold value for leverage.
        doa_attributes (LeverageDoa): Attributes of the leverage DOA.
    Properties:
        __name__ (str): Name of the DOA method.
        doa_matrix (np.array): Getter and setter for the DOA matrix.
        h_star (float): Getter and setter for the leverage threshold.
    Methods:
        __init__(): Initializes the Leverage class.
        __getitem__(key): Returns the key.
        calculate_threshold(): Calculates the leverage threshold (_h_star).
        calculate_matrix(): Calculates the DOA matrix (_doa_matrix) using the input data.
        fit(X: Union[np.array, pd.DataFrame]): Fits the model using the input data X.
        predict(new_data: Union[np.array, pd.DataFrame]) -> Iterable[Any]: Predicts if new data points are within DOA.
        _validate_input(data: Union[np.array, pd.DataFrame]): Validates and converts input data to numpy array if necessary.
        get_attributes(): Returns the attributes of the leverage DOA.
    """

    _doa = []
    _in_doa = []

    @property
    def __name__(self):
        return "LEVERAGE"

    def __init__(self) -> None:
        super().__init__()
        self._data: Union[np.array, pd.DataFrame] = None
        self._doa_matrix = None
        self._h_star = None
        self.doa_attributes = None

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
        self.doa_attributes = self.get_attributes()

    def predict(self, new_data: Union[np.array, pd.DataFrame]) -> Iterable[Any]:
        new_data = self._validate_input(new_data)
        doaAll = []
        for nd in new_data:
            d1 = np.dot(nd, self.doa_matrix)
            ndt = np.transpose(nd)
            d2 = np.dot(d1, ndt)
            if d2 < self._h_star:
                in_ad = True
            else:
                in_ad = False
            doa = {"h": d2, "hStar": self._h_star, "inDoa": in_ad}
            doaAll.append(doa)
        return doaAll

    def get_attributes(self):
        Leverage_data = LeverageDoa(h_star=self.h_star, doa_matrix=self.doa_matrix)
        return Leverage_data.to_dict()


class MeanVar(DOA):
    """Implements Mean and Variance domain of applicability .
    Initialized upon training data and holds the doa mean and the variance of the data.
    Calculates the mean and variance for a new instance of data or array of data and decides if in AD.
    """

    @property
    def __name__(self):
        return "MEAN_VAR"

    def __init__(self) -> None:
        super().__init__()
        self._data: np.array = None
        self.bounds = None
        self.doa_attributes = None

    def fit(self, X: np.array):
        X = self._validate_input(X)
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
        self.bounds = np.array(list_m_var)
        self.doa_attributes = self.get_attributes()

    def predict(self, new_data: np.array) -> Iterable[Any]:
        new_data = self._validate_input(new_data)
        doaAll = []
        in_doa = True
        for nd in new_data:
            for index, feature in enumerate(nd):
                bounds = self.bounds[index]
                bounds_data = [bounds[0] - 3 * bounds[1], bounds[0] + 3 * bounds[1]]
                if feature < bounds_data[0] or feature > bounds_data[1]:
                    in_doa = False
                    break
            out_of_doa_count = sum(
                1
                for feature in nd
                if feature < bounds_data[0] or feature > bounds_data[1]
            )
            out_of_doa_percentage = (out_of_doa_count / len(nd)) * 100
            doa = {"outOfDoaPercentage": out_of_doa_percentage, "inDoa": in_doa}
            doaAll.append(doa)
        return doaAll

    def get_attributes(self):
        return MeanVarDoa(bounds=self.bounds).to_dict()


class BoundingBox(DOA):
    @property
    def __name__(self):
        return "BOUNDING_BOX"

    def __init__(self) -> None:
        super().__init__()
        self._data: np.array = None
        self.bounding_box = None
        self.doa_attributes = None

    def fit(self, X: np.array):
        X = self._validate_input(X)
        self._data = X
        list_m_var = []
        for i in range(self._data.shape[1]):
            list_m_var.append([self._data[:, i].min(), self._data[:, i].max()])
        self.bounding_box = np.array(list_m_var)
        self.doa_attributes = self.get_attributes()

    def predict(self, new_data: np.array) -> Iterable[Any]:
        new_data = self._validate_input(new_data)
        doaAll = []
        in_doa = True
        for nd in new_data:
            for index, feature in enumerate(nd):
                bounds = self.bounding_box[index]
                bounds_data = [bounds[0], bounds[1]]
                if feature < bounds_data[0] or feature > bounds_data[1]:
                    in_doa = False
                    break
            out_of_doa_count = sum(
                1
                for feature in nd
                if feature < bounds_data[0] or feature > bounds_data[1]
            )
            out_of_doa_percentage = (out_of_doa_count / len(nd)) * 100
            doa = {"outOfDoaPercentage": out_of_doa_percentage, "inDoa": in_doa}
            doaAll.append(doa)
        return doaAll

    def get_attributes(self):
        return BoundingBoxDoa(bounding_box=self.bounding_box).to_dict()
