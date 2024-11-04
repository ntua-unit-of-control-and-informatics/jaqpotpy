# SklearnModel Documentation

## Class: `SklearnModel`

A class to represent a Scikit-learn model within the Jaqpot framework.

### Attributes

- `dataset (JaqpotpyDataset)`: The dataset used for training the model.
- `model (Any)`: The Scikit-learn model.
- `doa (Optional[Union[DOA, list]])`: Domain of Applicability methods.
- `preprocess_x (Optional[Union[BaseEstimator, List[BaseEstimator]]])`: Preprocessors for features.
- `preprocess_y (Optional[Union[BaseEstimator, List[BaseEstimator]]])`: Preprocessors for target variable.
- `random_seed (Optional[int])`: Random seed for reproducibility.

### Methods

#### `__init__(self, dataset: JaqpotpyDataset, model: Any, doa: Optional[Union[DOA, list]] = None, preprocess_x: Optional[Union[BaseEstimator, List[BaseEstimator]]] = None, preprocess_y: Optional[Union[BaseEstimator, List[BaseEstimator]]] = None, random_seed: Optional[int] = 1311)`

Initialize the SklearnModel.

**Args:**
- `dataset (JaqpotpyDataset)`: The dataset used for training the model.
- `model (Any)`: The Scikit-learn model.
- `doa (Optional[Union[DOA, list]])`: Domain of Applicability methods.
- `preprocess_x (Optional[Union[BaseEstimator, List[BaseEstimator]]])`: Preprocessors for features.
- `preprocess_y (Optional[Union[BaseEstimator, List[BaseEstimator]]])`: Preprocessors for target variable.
- `random_seed (Optional[int])`: Random seed for reproducibility.

#### `_dtypes_to_jaqpotypes(self)`

Convert dataset feature types to Jaqpot feature types.

#### `_extract_attributes(self, trained_class, trained_class_type)`

Extract attributes from a trained class.

**Args:**
- `trained_class`: The trained class instance.
- `trained_class_type (str)`: The type of the trained class.

**Returns:**
- `dict`: A dictionary of attributes.

#### `_add_class_to_extraconfig(self, added_class, added_class_type)`

Add a class to the extra configuration.

**Args:**
- `added_class`: The class to be added.
- `added_class_type (str)`: The type of the class.

#### `_map_onnx_dtype(self, dtype, shape=1)`

Map a data type to an ONNX data type.

**Args:**
- `dtype (str)`: The data type.
- `shape (int)`: The shape of the tensor.

**Returns:**
- ONNX data type.

#### `_create_onnx_preprocessor(self, onnx_options: Optional[Dict] = None)`

Create an ONNX preprocessor.

**Args:**
- `onnx_options (Optional[Dict])`: Options for ONNX conversion.

#### `_create_onnx_model(self, onnx_options: Optional[Dict] = None)`

Create an ONNX model.

**Args:**
- `onnx_options (Optional[Dict])`: Options for ONNX conversion.

#### `_labels_are_strings(self, y)`

Check if labels are strings.

**Args:**
- `y`: The labels.

**Returns:**
- `bool`: True if labels are strings, False otherwise.

#### `fit(self, onnx_options: Optional[Dict] = None)`

Fit the model to the dataset.

**Args:**
- `onnx_options (Optional[Dict])`: Options for ONNX conversion.

#### `predict(self, dataset: JaqpotpyDataset)`

Predict using the trained model.

**Args:**
- `dataset (JaqpotpyDataset)`: The dataset for prediction.

**Returns:**
- Predictions.

#### `_predict_with_X(self, X, model)`

Predict using the given model and features.

**Args:**
- `X`: The features.
- `model`: The model.

**Returns:**
- Predictions.

#### `predict_proba(self, dataset: JaqpotpyDataset)`

Predict probabilities using the trained model.

**Args:**
- `dataset (JaqpotpyDataset)`: The dataset for prediction.

**Returns:**
- List of probabilities.

#### `predict_onnx(self, dataset: JaqpotpyDataset)`

Predict using the ONNX model.

**Args:**
- `dataset (JaqpotpyDataset)`: The dataset for prediction.

**Returns:**
- ONNX predictions.

#### `predict_proba_onnx(self, dataset: JaqpotpyDataset)`

Predict probabilities using the ONNX model.

**Args:**
- `dataset (JaqpotpyDataset)`: The dataset for prediction.

**Returns:**
- List of ONNX probabilities.

#### `predict_doa(self, dataset: JaqpotpyDataset)`

Predict the Domain of Applicability (DOA).

**Args:**
- `dataset (JaqpotpyDataset)`: The dataset for prediction.

**Returns:**
- DOA results.

#### `deploy_on_jaqpot(self, jaqpot, name, description, visibility)`

Deploy the model on Jaqpot.

**Args:**
- `jaqpot`: The Jaqpot instance.
- `name (str)`: The name of the model.
- `description (str)`: The description of the model.
- `visibility`: The visibility of the model.

#### `_create_jaqpot_scores(self, fit_scores, score_type="train", n_output=1)`

Create Jaqpot scores.

**Args:**
- `fit_scores`: The fit scores.
- `score_type (str)`: The type of scores ('train', 'test', 'cross_validation').
- `n_output (int)`: The number of outputs.

#### `check_preprocessor(preprocessor_list: List, feat_type: str)`

Check if the preprocessors are valid.

**Args:**
- `preprocessor_list (List)`: The list of preprocessors.
- `feat_type (str)`: The type of features ('X' or 'y').

**Raises:**
- `ValueError`: If a preprocessor is not valid.

#### `cross_validate(self, dataset: JaqpotpyDataset, n_splits=5)`

Perform cross-validation.

**Args:**
- `dataset (JaqpotpyDataset)`: The dataset for cross-validation.
- `n_splits (int)`: The number of splits.

**Returns:**
- Cross-validation scores.

#### `_single_cross_validation(self, dataset: JaqpotpyDataset, y, n_splits=5, n_output=1)`

Perform a single cross-validation.

**Args:**
- `dataset (JaqpotpyDataset)`: The dataset for cross-validation.
- `y`: The target variable.
- `n_splits (int)`: The number of splits.
- `n_output (int)`: The number of outputs.

**Returns:**
- Average metrics.

#### `evaluate(self, dataset: JaqpotpyDataset)`

Evaluate the model on a dataset.

**Args:**
- `dataset (JaqpotpyDataset)`: The dataset for evaluation.

**Returns:**
- Evaluation scores.

#### `_evaluate_with_model(self, y_true, X_mat, model, output=1)`

Evaluate the model with given true values and features.

**Args:**
- `y_true`: The true values.
- `X_mat`: The features.
- `model`: The model.
- `output (int)`: The output index.

**Returns:**
- Evaluation metrics.

#### `randomization_test(self, train_dataset: JaqpotpyDataset, test_dataset: JaqpotpyDataset, n_iters=10)`

Perform a randomization test.

**Args:**
- `train_dataset (JaqpotpyDataset)`: The training dataset.
- `test_dataset (JaqpotpyDataset)`: The testing dataset.
- `n_iters (int)`: The number of iterations.

**Returns:**
- Randomization test results.

#### `_get_metrics(self, y_true, y_pred)`

Get metrics based on the task type.

**Args:**
- `y_true`: The true values.
- `y_pred`: The predicted values.

**Returns:**
- Metrics.

#### `_get_classification_metrics(y_true, y_pred, binary=True)`

Get classification metrics.

**Args:**
- `y_true`: The true values.
- `y_pred`: The predicted values.
- `binary (bool)`: Whether the classification is binary.

**Returns:**
- Classification metrics.

#### `_get_regression_metrics(y_true, y_pred)`

Get regression metrics.

**Args:**
- `y_true`: The true values.
- `y_pred`: The predicted values.

**Returns:**
- Regression metrics.