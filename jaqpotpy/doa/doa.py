from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Iterable, Any, Union

from jaqpotpy.api.openapi.models.bounding_box_doa import BoundingBoxDoa
from jaqpotpy.api.openapi.models.leverage_doa import LeverageDoa


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
    """
    Leverage class for Domain of Applicability (DOA) analysis.
    This class implements the Leverage method for determining the domain of applicability
    of a given dataset. It provides methods to fit the model, predict new data points,
    and calculate the necessary matrices and thresholds.
    Attributes:
        _doa (list): List to store leverage values.
        _in_doa (list): List to store boolean values indicating if data points are within the domain of applicability.
        _data (Union[np.array, pd.DataFrame]): The input data used for fitting the model.
        doa_matrix (np.array): The matrix used for leverage calculations.
        _h_star (float): The threshold value for determining if a data point is within the domain of applicability.
        doa_attributes (LeverageDoa): Object containing the attributes of the leverage model.
    Methods:
        __name__: Returns the name of the method ("LEVERAGE").
        __init__(): Initializes the Leverage object.
        __getitem__(key): Returns the key.
        doa_matrix: Property to get and set the doa_matrix attribute.
        h_star: Property to get and set the _h_star attribute.
        calculate_threshold(): Calculates the threshold value (_h_star) based on the input data.
        calculate_matrix(): Calculates the leverage matrix (doa_matrix) using the input data.
        fit(X): Fits the model using the input data X.
        predict(new_data): Predicts whether new data points are within the domain of applicability.
        _validate_input(data): Validates and converts the input data to a numpy array if it is a pandas DataFrame.
        get_attributes(): Returns the attributes of the leverage model as a LeverageDoa object.
    """

    _doa = []
    _in_doa = []

    @property
    def __name__(self):
        return "LEVERAGE"

    def __init__(self) -> None:
        # self._scaler: BaseEstimator = scaler
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
            doa = {"h": d2, "h_star": self._h_star, "in_doa": in_ad}
            doaAll.append(doa)
        return doaAll

    def _validate_input(self, data: Union[np.array, pd.DataFrame]):
        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
        else:
            return data

    def get_attributes(self):
        Leverage_data = LeverageDoa(h_star=self.h_star, doa_matrix=self.doa_matrix)
        return Leverage_data


class MeanVar(DOA):
    """Implements Mean and Variance domain of applicability .
    Initialized upon training data and holds the doa mean and the variance of the data.
    Calculates the mean and variance for a new instance of data or array of data and decides if in AD.
    """

    _doa = []
    _in_doa = []

    @property
    def __name__(self):
        return "MEAN_VAR"

    def __init__(self) -> None:
        self._data: np.array = None
        self.bounds = None
        self.doa_attributes = None

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
        self.bounds = np.array(list_m_var)
        self.doa_attributes = self.get_attributes()

    def predict(self, new_data: np.array) -> Iterable[Any]:
        doaAll = []
        self._doa = []
        self._in_doa = []
        in_doa = True
        for nd in new_data:
            for index, row in enumerate(nd):
                bounds = self.bounds[index]
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
        return {"bounds": self.bounds}


class BoundingBox(DOA):
    _doa = []
    _in_doa = []

    @property
    def __name__(self):
        return "BOUNDING_BOX"

    def __init__(self) -> None:
        self._data: np.array = None
        self.bounding_box = None
        self.doa_attributes = None

    def fit(self, X: np.array):
        self._data = X
        list_m_var = []
        # if isinstance(self._data, pd.DataFrame):
        #     self._data = self._data.to_numpy()
        # for i in range(self._data.shape[1]):
        #     list_m_var.append([np.min(self._data[:, i]), np.max(self._data[:, i])])
        for i in range(self._data.shape[1]):
            list_m_var.append(
                [self._data.iloc[:, i].min(), self._data.iloc[:, i].max()]
            )
        self.bounding_box = np.array(list_m_var)
        self.doa_attributes = self.get_attributes()

    def predict(self, new_data: np.array) -> Iterable[Any]:
        new_data = self._validate_input(new_data)
        doaAll = []
        self._doa = []
        self._in_doa = []
        in_doa = True
        for nd in new_data:
            for index, row in enumerate(nd):
                bounds = self.bounding_box[index]
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

    def _validate_input(self, data: Union[np.array, pd.DataFrame]):
        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
        else:
            return data

    def get_attributes(self):
        return BoundingBoxDoa(bounding_box=self.bounding_box)
