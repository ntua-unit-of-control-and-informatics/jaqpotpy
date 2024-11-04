# Model Class Documentation

## Class: `Model`

Base class for all models in jaqpotpy.

### Attributes

- **_model** (`Any`): The underlying model.
- **_doa** (`DOA`): Domain of Applicability object.
- **_descriptors** (`MolecularFeaturizer`): Molecular featurizer.
- **_X** (`Iterable[str]`): Input features.
- **_Y** (`Iterable[str]`): Output features.
- **_X_indices** (`Iterable[int]`): Indices of input features.
- **_prediction** (`Any`): Model predictions.
- **_probability** (`Any`): Prediction probabilities.
- **_external** (`Any`): External data.
- **_smiles** (`Any`): SMILES representation of molecules.
- **_external_feats** (`Iterable[str]`): External features.
- **_model_title** (`Any`): Title of the model.
- **_modeling_task** (`Any`): Description of the modeling task.
- **_library** (`Iterable[str]`): Library used.
- **_version** (`Iterable[str]`): Version of the model.
- **_jaqpotpy_version** (`Any`): Version of jaqpotpy.
- **_jaqpotpy_docker** (`Any`): Docker information for jaqpotpy.
- **_optimizer** (`Any`): Optimizer used.

### Properties

- **smiles**: Get or set the SMILES representation of molecules.
- **descriptors**: Get or set the molecular featurizer.
- **doa**: Get or set the Domain of Applicability object.
- **X**: Get or set the input features.
- **Y**: Get or set the output features.
- **external_feats**: Get or set the external features.
- **model**: Get or set the underlying model.
- **model_title**: Get or set the title of the model.
- **prediction**: Get or set the model predictions.
- **probability**: Get or set the prediction probabilities.
- **library**: Get or set the library used.
- **optimizer**: Get or set the optimizer used.
- **version**: Get or set the version of the model.
- **jaqpotpy_version**: Get or set the version of jaqpotpy.
- **modeling_task**: Get or set the description of the modeling task.
- **jaqpotpy_docker**: Get or set the Docker information for jaqpotpy.

### Methods

- **fit**: Fit the model to the data.
  ```python
  def fit(self):
      """Fit the model to the data."""
      raise NotImplementedError("Not implemented")
- predict: Predict using the model.
  ```python
    def predict(self, X):
        """Predict using the model.

        Args:
            X: Input data for prediction.

        Returns:
            Predictions for the input data.
        """
        raise NotImplementedError("Not implemented")