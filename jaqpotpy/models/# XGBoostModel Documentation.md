# XGBoostModel Documentation

## Class: XGBoostModel

XGBoostModel class for handling XGBoost models within the Jaqpotpy framework.

### Attributes:
- `dataset (JaqpotpyDataset)`: The dataset used for training the model.
- `model (Any)`: The XGBoost model instance.
- `doa (Optional[DOA or list])`: Domain of Applicability (DOA) methods.
- `preprocess_x (Optional[Union[BaseEstimator, List[BaseEstimator]]])`: Preprocessing steps for input features.
- `preprocess_y (Optional[Union[BaseEstimator, List[BaseEstimator]]])`: Preprocessing steps for target features.

## Method: `__init__`

Initializes the XGBoostModel with the given dataset, model, and optional preprocessing steps.

### Args:
- `dataset (JaqpotpyDataset)`: The dataset used for training the model.
- `model (Any)`: The XGBoost model instance.
- `doa (Optional[DOA or list])`: Domain of Applicability (DOA) methods.
- `preprocess_x (Optional[Union[BaseEstimator, List[BaseEstimator]]])`: Preprocessing steps for input features.
- `preprocess_y (Optional[Union[BaseEstimator, List[BaseEstimator]]])`: Preprocessing steps for target features.

## Method: `_create_onnx`

Creates an ONNX representation of the trained model.

### Args:
- `onnx_options (Optional[Dict])`: Additional options for ONNX conversion.

## Method: `_convert_regressor`

Registers the XGBRegressor model for ONNX conversion.

## Method: `_convert_classifier`

Registers the XGBClassifier model for ONNX conversion.