# Domain of Applicability (DOA) Module

This module provides classes for calculating the Domain of Applicability (DOA) using different methods. The available methods are Leverage, Mean-Variance, and Bounding Box.

## Classes

### DOA (Abstract Base Class)

#### Description
Abstract class for DOA methods.

#### Attributes
- `_in_doa` (list): List to store boolean values indicating if data points are within DOA.
- `_data` (Union[np.array, pd.DataFrame]): Input data used for DOA calculation.

#### Properties
- `__name__` (str): Name of the DOA method.
- `in_doa` (list): Getter and setter for the `_in_doa` attribute.
- `data` (Union[np.array, pd.DataFrame]): Getter and setter for the `_data` attribute.

#### Methods
- `fit(X: np.array)`: Abstract method to fit the model using the input data X.
- `predict(data: Iterable[Any]) -> Iterable[Any]`: Abstract method to predict if data points are within DOA.
- `get_attributes()`: Abstract method to get the attributes of the DOA.
- `_validate_input(data: Union[np.array, pd.DataFrame])`: Validates and converts input data to numpy array if necessary.

### Leverage

#### Description
Leverage class for DOA calculation using the leverage method.

#### Attributes
- `_doa` (list): List to store leverage values.
- `_in_doa` (list): List to store boolean values indicating if data points are within DOA.
- `_data` (Union[np.array, pd.DataFrame]): Input data used for DOA calculation.
- `_doa_matrix` (np.array): Matrix used for leverage calculation.
- `_h_star` (float): Threshold value for leverage.
- `doa_attributes` (LeverageDoa): Attributes of the leverage DOA.

#### Properties
- `__name__` (str): Name of the DOA method.
- `doa_matrix` (np.array): Getter and setter for the DOA matrix.
- `h_star` (float): Getter and setter for the leverage threshold.

#### Methods
- `__init__()`: Initializes the Leverage class.
- `__getitem__(key)`: Returns the key.
- `calculate_threshold()`: Calculates the leverage threshold (`_h_star`).
- `calculate_matrix()`: Calculates the DOA matrix (`_doa_matrix`) using the input data.
- `fit(X: Union[np.array, pd.DataFrame])`: Fits the model using the input data X.
- `predict(new_data: Union[np.array, pd.DataFrame]) -> Iterable[Any]`: Predicts if new data points are within DOA.
- `get_attributes()`: Returns the attributes of the leverage DOA.

### MeanVar

#### Description
Implements Mean and Variance domain of applicability. Initialized upon training data and holds the DOA mean and the variance of the data.

#### Attributes
- `_data` (np.array): Input data used for DOA calculation.
- `bounds` (np.array): Array containing the mean, standard deviation, and variance for each feature.
- `doa_attributes` (MeanVarDoa): Attributes of the mean-variance DOA.

#### Properties
- `__name__` (str): Name of the DOA method.

#### Methods
- `__init__()`: Initializes the MeanVar class.
- `fit(X: np.array)`: Fits the model using the input data X.
- `predict(new_data: np.array) -> Iterable[Any]`: Predicts if new data points are within DOA.
- `get_attributes()`: Returns the attributes of the mean-variance DOA.

### BoundingBox

#### Description
BoundingBox class for DOA calculation using the bounding box method.

#### Attributes
- `_data` (np.array): Input data used for DOA calculation.
- `bounding_box` (np.array): Array containing the min and max bounds for each feature.
- `doa_attributes` (BoundingBoxDoa): Attributes of the bounding box DOA.

#### Properties
- `__name__` (str): Name of the DOA method.

#### Methods
- `__init__()`: Initializes the BoundingBox class.
- `fit(X: np.array)`: Fits the model using the input data X.
- `predict(new_data: np.array) -> Iterable[Any]`: Predicts if new data points are within DOA.
- `get_attributes()`: Returns the attributes of the bounding box DOA.