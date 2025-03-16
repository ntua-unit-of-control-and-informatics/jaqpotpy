from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Union, Iterable, Any, Callable
from scipy.stats import chi2
from scipy.spatial.distance import pdist, cdist
from scipy.linalg import pinv

from jaqpot_api_client.models.bounding_box_doa import BoundingBoxDoa
from jaqpot_api_client.models.leverage_doa import LeverageDoa
from jaqpot_api_client.models.mean_var_doa import MeanVarDoa
from jaqpot_api_client.models.mahalanobis_doa import MahalanobisDoa
from jaqpot_api_client.models.kernel_based_doa import KernelBasedDoa
from jaqpot_api_client.models.city_block_doa import CityBlockDoa


class DOA(ABC):
    """Abstract class for Domain of Applicability (DOA) methods.

    Attributes:
        _in_doa (list): List to store boolean values indicating if data points are within DOA.

        _data (Union[np.array, pd.DataFrame]): Input data used for DOA calculation.

    Properties:
        __name__ (str): Name of the DOA method.

        in_doa (list): Getter and setter for the in_doa attribute.

        data (Union[np.array, pd.DataFrame]): Getter and setter for the data attribute.

    Methods:
        fit(X: np.array): Abstract method to fit the model using the input data X.
        predict(data: Iterable[Any]) -> Iterable[Any]: Abstract method to predict if data points are within DOA.
        get_attributes(): Abstract method to get the attributes of the DOA.
        _validate_input(data: Union[np.array, pd.DataFrame]): Validates and converts input data to numpy array if necessary.
    """

    @property
    def __name__(self):
        """Name of the DOA method."""
        return NotImplementedError

    @property
    def in_doa(self):
        """Getter for the in_doa attribute."""
        return self._in_doa

    @in_doa.setter
    def in_doa(self, value):
        """Setter for the in_doa attribute."""
        self._in_doa = value

    @property
    def data(self):
        """Getter for the data attribute."""
        return self._data

    @data.setter
    def data(self, value):
        """Setter for the data attribute."""
        self._data = value

    @abstractmethod
    def fit(self, X: np.array):
        """Abstract method to fit the model using the input data X."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: Iterable[Any]) -> Iterable[Any]:
        """Abstract method to predict if data points are within DOA."""
        raise NotImplementedError

    @abstractmethod
    def get_attributes(self):
        """Abstract method to get the attributes of the DOA."""
        raise NotImplementedError

    def _validate_input(self, data: Union[np.array, pd.DataFrame]):
        """Validates and converts input data to numpy array if necessary.
        Args:
            data (Union[np.array, pd.DataFrame]): Input data.
        Returns:
            np.array: Validated input data as numpy array.
        """
        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Input must be a NumPy array or Pandas DataFrame")


class Leverage(DOA):
    """Leverage class for Domain of Applicability (DOA) calculation using the leverage method.

    Attributes:
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

    @property
    def __name__(self):
        """Name of the DOA method."""
        return "LEVERAGE"

    def __init__(self) -> None:
        """Initializes the Leverage class."""
        super().__init__()
        self._data: Union[np.array, pd.DataFrame] = None
        self._doa_matrix = None
        self._h_star = None
        self.doa_attributes = None

    def __getitem__(self, key):
        """Returns the key."""
        return key

    @property
    def doa_matrix(self):
        """Getter for the DOA matrix."""
        return self._doa_matrix

    @doa_matrix.setter
    def doa_matrix(self, value):
        """Setter for the DOA matrix."""
        self._doa_matrix = value

    @property
    def h_star(self):
        """Getter for the leverage threshold."""
        return self._h_star

    @h_star.setter
    def h_star(self, value):
        """Setter for the leverage threshold."""
        self._h_star = value

    def calculate_threshold(self):
        """Calculates the leverage threshold (_h_star)."""
        shape = self._data.shape
        h_star = (3 * (shape[1] + 1)) / shape[0]
        self._h_star = h_star

    def calculate_matrix(self):
        """Calculates the DOA matrix (_doa_matrix) using the input data."""
        x_T = self._data.transpose()
        x_out = x_T.dot(self._data).astype(np.float64)
        lambda_reg = 1e-10
        x_out += np.eye(x_out.shape[0]) * lambda_reg
        self.doa_matrix = None

        while self.doa_matrix is None:
            try:
                self._doa_matrix = pinv(x_out)
            except:
                print(
                    "Matrix inversion failed, trying with reduce ridge lambda. Current value: ",
                    lambda_reg,
                )
                lambda_reg = lambda_reg * 10
                x_out += np.eye(x_out.shape[0]) * lambda_reg

    def fit(self, X: Union[np.array, pd.DataFrame]):
        """Fits the model using the input data X.
        Args:
            X (Union[np.array, pd.DataFrame]): Input data.
        """
        self._data = self._validate_input(X)
        self.calculate_matrix()
        self.calculate_threshold()
        self.doa_attributes = self.get_attributes()

    def predict(self, new_data: Union[np.array, pd.DataFrame]) -> Iterable[Any]:
        """Predicts if new data points are within DOA.
        Args:
            new_data (Union[np.array, pd.DataFrame]): New data points to be predicted.
        Returns:
            Iterable[Any]: List of dictionaries containing the leverage value, threshold, and a boolean indicating if the data point is within DOA.
        """
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
        """Returns the attributes of the leverage DOA.
        Returns:
            LeverageDoa: Attributes of the leverage DOA.
        """
        Leverage_data = LeverageDoa(h_star=self.h_star, doa_matrix=self.doa_matrix)
        return Leverage_data.to_dict()


class MeanVar(DOA):
    """Implements Mean and Variance domain of applicability.

    Initialized upon training data and holds the DOA mean and the variance of the data.
    Calculates the mean and variance for a new instance of data or array of data and decides if in AD.

    Attributes:
        _data (np.array): Input data used for DOA calculation.
        bounds (np.array): Array containing the mean, standard deviation, and variance for each feature.
        doa_attributes (MeanVarDoa): Attributes of the mean-variance DOA.

    Properties:
        __name__ (str): Name of the DOA method.

    Methods:
        __init__(): Initializes the MeanVar class.
        fit(X: np.array): Fits the model using the input data X.
        predict(new_data: np.array) -> Iterable[Any]: Predicts if new data points are within DOA.
        _validate_input(data: Union[np.array, pd.DataFrame]): Validates and converts input data to numpy array if necessary.
        get_attributes(): Returns the attributes of the mean-variance DOA.
    """

    @property
    def __name__(self):
        """Name of the DOA method."""
        return "MEAN_VAR"

    def __init__(self) -> None:
        """Initializes the MeanVar class."""
        super().__init__()
        self._data: np.array = None
        self.bounds = None
        self.doa_attributes = None

    def fit(self, X: np.array):
        """Fits the model using the input data X.
        Calculates the mean, standard deviation, and variance for each feature in the input data.
        Args:
            X (np.array): Input data.
        """
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
        """Predicts if new data points are within DOA.
        Args:
            new_data (np.array): New data points to be predicted.
        Returns:
            Iterable[Any]: List of dictionaries containing the percentage of features out of DOA and a boolean indicating if the data point is within DOA.
        """
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
    """BoundingBox class for Domain of Applicability (DOA) calculation using the bounding box method.

    Attributes:
        _data (np.array): Input data used for DOA calculation.
        bounding_box (np.array): Array containing the min and max bounds for each feature.
        doa_attributes (BoundingBoxDoa): Attributes of the bounding box DOA.

    Properties:
        __name__ (str): Name of the DOA method.

    Methods:
        __init__(): Initializes the BoundingBox class.
        fit(X: np.array): Fits the model using the input data X.
        predict(new_data: np.array) -> Iterable[Any]: Predicts if new data points are within DOA.
        _validate_input(data: Union[np.array, pd.DataFrame]): Validates and converts input data to numpy array if necessary.
        get_attributes(): Returns the attributes of the bounding box DOA.
    """

    @property
    def __name__(self):
        return "BOUNDING_BOX"

    def __init__(self) -> None:
        super().__init__()
        self._data: np.array = None
        self.bounding_box = None
        self.doa_attributes = None

    def fit(self, X: np.array):
        """
        Fits the model using the input data X.
        Calculates the min and max bounds for each feature in the input data.
        Args:
            X (np.array): Input data.
        """
        X = self._validate_input(X)
        self._data = X
        list_m_var = []
        for i in range(self._data.shape[1]):
            list_m_var.append([self._data[:, i].min(), self._data[:, i].max()])
        self.bounding_box = np.array(list_m_var)
        self.doa_attributes = self.get_attributes()

    def predict(self, new_data: np.array) -> Iterable[Any]:
        """
        Predicts if new data points are within DOA.
        Args:
            new_data (np.array): New data points to be predicted.
        Returns:
            Iterable[Any]: List of dictionaries containing the percentage of features out of DOA and a boolean indicating if the data point is within DOA.
        """
        new_data = self._validate_input(new_data)
        doaAll = []
        for nd in new_data:
            out_of_doa_count = 0
            for index, feature in enumerate(nd):
                bounds = self.bounding_box[index]
                if feature < bounds[0] or feature > bounds[1]:
                    out_of_doa_count += 1
            out_of_doa_percentage = (out_of_doa_count / len(nd)) * 100
            in_doa = True if out_of_doa_count == 0 else False
            doa = {"outOfDoaPercentage": out_of_doa_percentage, "inDoa": in_doa}
            doaAll.append(doa)
        return doaAll

    def get_attributes(self):
        return BoundingBoxDoa(bounding_box=self.bounding_box).to_dict()


class Mahalanobis(DOA):
    """Mahalanobis Distance Domain of Applicability (DOA) calculation class.

    Attributes:
        _data (Union[np.array, pd.DataFrame]): Input data used for DOA calculation.
        _mean_vector (np.array): Mean vector of the training data.
        _cov_matrix (np.array): Covariance matrix of the training data.
        _inv_cov_matrix (np.array): Inverse of the covariance matrix.
        _threshold (float): Threshold value for Mahalanobis distance.
        doa_attributes (MahalanobisDoA): Attributes of the Mahalanobis DOA.

    Methods:
        __init__(): Initializes the Mahalanobis DOA class.
        fit(X: Union[np.array, pd.DataFrame]): Fits the model using the input data.
        predict(new_data: Union[np.array, pd.DataFrame]) -> Iterable[Any]: Predicts if new data points are within DOA.
        calculate_distance(sample: np.array) -> float: Calculates Mahalanobis distance for a sample.
        calculate_threshold(): Calculates the Mahalanobis distance threshold.
    """

    @property
    def __name__(self):
        """Name of the DOA method."""
        return "MAHALANOBIS"

    def __init__(self, chi2_quantile=0.95) -> None:
        """Initializes the Mahalanobis DOA class."""
        super().__init__()
        if chi2_quantile < 0 or chi2_quantile > 1:
            raise ValueError(
                "The chi-squared quantile cannot be less than 0 or greater than 1."
            )

        self.chi2_quantile = chi2_quantile
        self._data = None
        self._mean_vector = None
        self._cov_matrix = None
        self._inv_cov_matrix = None
        self._threshold = None
        self.doa_attributes = None

    def fit(self, X: Union[np.array, pd.DataFrame]):
        """
        Fits the model using the input data.

        Args:
            X (Union[np.array, pd.DataFrame]): Input training data.
        """
        self._data = self._validate_input(X)

        self._mean_vector = np.mean(self._data, axis=0)
        self._cov_matrix = np.cov(self._data, rowvar=False)
        epsilon = 1e-6
        self._cov_matrix += np.eye(self._cov_matrix.shape[0]) * epsilon
        self._inv_cov_matrix = np.linalg.inv(self._cov_matrix)
        self.calculate_threshold()
        self.doa_attributes = self.get_attributes()

    def calculate_distance(self, sample: np.array) -> float:
        """
        Calculates Mahalanobis distance for a sample.

        Args:
            sample (np.array): Input sample to calculate distance for.

        Returns:
            float: Mahalanobis distance of the sample.
        """
        diff = sample - self._mean_vector
        mahalanobis_dist = np.sqrt(diff.dot(self._inv_cov_matrix).dot(diff))
        return mahalanobis_dist

    def calculate_threshold(self):
        """
        Calculates the Mahalanobis distance threshold using chi-square distribution.
        Uses 99% confidence level by default.
        """
        self._threshold = np.sqrt(
            chi2.ppf(self.chi2_quantile, df=(self._data.shape[1] - 1))
        )

    def predict(self, new_data: Union[np.array, pd.DataFrame]) -> Iterable[Any]:
        """
        Predicts if new data points are within DOA.

        Args:
            new_data (Union[np.array, pd.DataFrame]): New data points to be predicted.

        Returns:
            Iterable[Any]: List of dictionaries containing the Mahalanobis distance,
            threshold, and a boolean indicating if the data point is within DOA.
        """
        new_data = self._validate_input(new_data)
        doa_results = []

        for nd in new_data:
            distance = self.calculate_distance(nd)
            in_ad = False
            if distance <= self._threshold:
                in_ad = True

            doa = {
                "mahalanobisDistance": distance,
                "threshold": self._threshold,
                "inDoa": in_ad,
            }
            doa_results.append(doa)

        return doa_results

    def get_attributes(self):
        """
        Returns the attributes of the Mahalanobis DOA.

        Returns:
            MahalanobisDOAAttributes: Attributes of the Mahalanobis DOA.
        """
        mahalanobis_data = MahalanobisDoa(
            mean_vector=self._mean_vector,
            inv_cov_matrix=self._inv_cov_matrix,
            threshold=self._threshold,
        )
        return mahalanobis_data.to_dict()


class KernelBased(DOA):
    """Enhanced Kernel-based Distance of Applicability (DOA) calculation class.

    Supports multiple kernel types and more flexible threshold calculation.
    """

    @property
    def __name__(self):
        """Name of the DOA method."""
        return "KERNEL_BASED"

    def __init__(
        self,
        kernel_type="GAUSSIAN",
        threshold_method="percentile",
        threshold_percentile=5,
        sigma=None,
        gamma=None,
    ):
        """
        Initialize the Kernel DOA with configurable parameters.

        Args:
            kernel_type (str): Type of kernel to use.
                Must be 'gaussian', 'rbf', or 'laplacian'.
            threshold_method (str): Method for threshold calculation.
                Must be 'percentile' or 'mean_std'.
            threshold_percentile (float): Percentile for threshold if using
                percentile method.
            sigma (float, optional): Kernel width parameter. Must be None or positive.

            gamma (float, optional): rbf gamma

        Raises:
            ValueError: If parameters do not meet specified constraints.
        """
        valid_kernel_types = ["GAUSSIAN", "RBF", "LAPLACIAN"]
        if kernel_type not in valid_kernel_types:
            raise ValueError(
                f"Invalid kernel type. Must be one of {valid_kernel_types}"
            )
        self._kernel_type = kernel_type

        valid_threshold_methods = ["percentile", "mean_std"]
        if threshold_method not in valid_threshold_methods:
            raise ValueError(
                f"Invalid threshold method. Must be one of {valid_threshold_methods}"
            )
        self._threshold_method = threshold_method

        if not (0 < threshold_percentile < 100):
            raise ValueError("Threshold percentile must be between 0 and 100")
        self._threshold_percentile = threshold_percentile

        if sigma is not None and sigma <= 0:
            raise ValueError("Sigma must be None or a positive number")
        self._sigma = sigma
        self._gamma = gamma

        self._data = None
        self._kernel_distances = None
        self._threshold = None
        self.doa_attributes = None

    def _select_kernel(self) -> Callable:
        """
        Select and return the appropriate kernel function.

        Returns:
            Callable: Kernel distance calculation function.
        """
        kernel_map = {
            "gaussian": self._gaussian_kernel,
            "rbf": self._rbf_kernel,
            "laplacian": self._laplacian_kernel,
        }
        return kernel_map.get(self._kernel_type, self._gaussian_kernel)

    def _gaussian_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Calculate Gaussian kernel distances using Euclidean distance.

        Args:
            X (np.ndarray): First set of points.
            Y (np.ndarray): Second set of points.
            sigma (float, optional): Kernel width parameter.

        Returns:
            np.ndarray: Gaussian kernel distance matrix.
        """

        dist_matrix = cdist(X, Y, metric="euclidean")
        return np.exp(-(dist_matrix**2) / (2 * self._sigma**2))

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Radial Basis Function kernel calculation with custom gamma.

        Args:
            X (np.ndarray): First set of points.
            Y (np.ndarray): Second set of points.
            sigma (float, optional): Kernel width parameter.

        Returns:
            np.ndarray: RBF kernel matrix.
        """
        gamma = self._gamma if self._gamma is not None else 1 / (2 * self._sigma**2)
        dist_matrix = cdist(X, Y, metric="euclidean")
        return np.exp(-gamma * dist_matrix**2)

    def _laplacian_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Laplacian kernel calculation using Manhattan distance.

        Args:
            X (np.ndarray): First set of points.
            Y (np.ndarray): Second set of points.
            sigma (float, optional): Kernel width parameter.

        Returns:
            np.ndarray: Laplacian kernel matrix.
        """
        dist_matrix = cdist(X, Y, metric="cityblock")
        return np.exp(-dist_matrix / self._sigma)

    def _calculate_threshold(self, kernel_distances: np.ndarray) -> float:
        """
        Calculate threshold based on selected method.

        Args:
            kernel_distances (np.ndarray): Kernel distance matrix.

        Returns:
            float: Calculated threshold value.
        """
        if self._threshold_method == "percentile":
            return np.percentile(kernel_distances, self._threshold_percentile)
        elif self._threshold_method == "mean_std":
            return np.mean(kernel_distances) + 2 * np.std(kernel_distances)
        else:
            return np.mean(kernel_distances) * 3  # Default fallback

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Fit the kernel DOA model.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Training data.
        """
        self._data = self._validate_input(X)
        if self._sigma is None:
            self._sigma = np.median(pdist(self._data))
        # Select and apply kernel function
        kernel_func = self._select_kernel()
        self._kernel_distances = kernel_func(self._data, self._data)

        # Calculate threshold
        self._threshold = self._calculate_threshold(self._kernel_distances)
        self.doa_attributes = self.get_attributes()

    def predict(self, new_data: Union[np.ndarray, pd.DataFrame]) -> Iterable[dict]:
        """
        Predict DOA for new data points.

        Args:
            new_data (Union[np.ndarray, pd.DataFrame]): Data to predict.

        Returns:
            Iterable[dict]: Prediction results for each data point.
        """
        new_data = self._validate_input(new_data)
        kernel_func = self._select_kernel()

        doa_results = []
        if isinstance(self._data, list):
            self._data = np.array(self._data)  # Convert to NumPy array
        for point in new_data:
            # Compute kernel distances between the new point and all training data points
            point_distances = [
                kernel_func(point.reshape(1, -1), self._data[i].reshape(1, -1))[0, 0]
                for i in range(len(self._data))
            ]

            # Calculate average kernel distance
            avg_distance = np.mean(point_distances)
            in_ad = False
            if avg_distance >= self._threshold:
                in_ad = True
            doa_results.append(
                {
                    "kernelDistance": avg_distance,
                    "threshold": self._threshold,
                    "inDoa": in_ad,
                }
            )

        return doa_results

    def get_attributes(self):
        """
        Returns the attributes of the Kernel DOA.

        Returns:
            KernelDOAAttributes: Attributes of the Kernel DOA.
        """
        kernel_data = KernelBasedDoa(
            sigma=self._sigma,
            gamma=self._gamma,
            threshold=self._threshold,
            kernel_type=self._kernel_type,
            data_points=self._data,
        )
        return kernel_data.to_dict()


class CityBlock(DOA):
    """City Block (Manhattan) Distance Domain of Applicability (DOA) calculation class.

    Attributes:
        _data (Union[np.array, pd.DataFrame]): Input data used for DOA calculation.
        _mean_vector (np.array): Mean vector of the training data.
        _threshold (float): Threshold value for City Block distance.
        doa_attributes (CityBlockDoA): Attributes of the City Block DOA.

    Methods:
        __init__(): Initializes the CityBlock DOA class.
        fit(X: Union[np.array, pd.DataFrame]): Fits the model using the input data.
        predict(new_data: Union[np.array, pd.DataFrame]) -> Iterable[Any]: Predicts if new data points are within DOA.
        calculate_distance(sample: np.array) -> float: Calculates City Block distance for a sample.
        calculate_threshold(): Calculates the City Block distance threshold.
    """

    @property
    def __name__(self):
        """Name of the DOA method."""
        return "CITY_BLOCK"

    def __init__(self, threshold_percentile=95) -> None:
        """Initializes the CityBlock DOA class."""
        super().__init__()
        if not (0 < threshold_percentile < 100):
            raise ValueError("The threshold percentile must be between 0 and 100.")

        self.threshold_percentile = threshold_percentile
        self._data = None
        self._mean_vector = None
        self._threshold = None
        self.doa_attributes = None

    def fit(self, X: Union[np.array, pd.DataFrame]):
        """
        Fits the model using the input data.

        Args:
            X (Union[np.array, pd.DataFrame]): Input training data.
        """
        self._data = self._validate_input(X)

        # Compute the mean vector of the training data
        self._mean_vector = np.mean(self._data, axis=0)
        self.calculate_threshold()
        self.doa_attributes = self.get_attributes()

    def calculate_distance(self, sample: np.array) -> float:
        """
        Calculates City Block (Manhattan) distance for a sample.

        Args:
            sample (np.array): Input sample to calculate distance for.

        Returns:
            float: City Block distance of the sample.
        """
        return np.sum(np.abs(sample - self._mean_vector))

    def calculate_threshold(self):
        """
        Calculates the City Block distance threshold based on the chosen percentile.
        """
        distances = np.array([self.calculate_distance(sample) for sample in self._data])
        self._threshold = np.percentile(distances, self.threshold_percentile)

    def predict(self, new_data: Union[np.array, pd.DataFrame]) -> Iterable[Any]:
        """
        Predicts if new data points are within DOA.

        Args:
            new_data (Union[np.array, pd.DataFrame]): New data points to be predicted.

        Returns:
            Iterable[Any]: List of dictionaries containing the City Block distance,
            threshold, and a boolean indicating if the data point is within DOA.
        """
        new_data = self._validate_input(new_data)
        doa_results = []

        for nd in new_data:
            distance = self.calculate_distance(nd)
            in_ad = False
            if distance <= self._threshold:
                in_ad = True

            doa = {
                "cityBlockDistance": distance,
                "threshold": self._threshold,
                "inDoa": in_ad,
            }
            doa_results.append(doa)

        return doa_results

    def get_attributes(self):
        """
        Returns the attributes of the CityBlock DOA.

        Returns:
            CityBlockDOAAttributes: Attributes of the CityBlock DOA.
        """
        cityblock_data = CityBlockDoa(
            mean_vector=self._mean_vector,
            threshold=self._threshold,
        )
        return cityblock_data.to_dict()
