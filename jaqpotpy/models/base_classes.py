from jaqpotpy.doa import DOA
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from typing import Any, Iterable


class Model(object):
    """
    Base class for all models in jaqpotpy.

    Attributes:
        _model (Any): The underlying model.
        _doa (DOA): Domain of Applicability object.
        _descriptors (MolecularFeaturizer): Molecular featurizer.
        _X (Iterable[str]): Input features.
        _Y (Iterable[str]): Output features.
        _X_indices (Iterable[int]): Indices of input features.
        _prediction (Any): Model predictions.
        _probability (Any): Prediction probabilities.
        _external (Any): External data.
        _smiles (Any): SMILES representation of molecules.
        _external_feats (Iterable[str]): External features.
        _model_title (Any): Title of the model.
        _modeling_task (Any): Description of the modeling task.
        _library (Iterable[str]): Library used.
        _version (Iterable[str]): Version of the model.
        _jaqpotpy_version (Any): Version of jaqpotpy.
        _jaqpotpy_docker (Any): Docker information for jaqpotpy.
        _optimizer (Any): Optimizer used.
    """

    _model: Any
    _doa: DOA
    _descriptors: MolecularFeaturizer
    _X: Iterable[str]
    _Y: Iterable[str]
    _X_indices: Iterable[int]
    _prediction: Any
    _probability = None
    _external = None
    _smiles = None
    _external_feats = None
    _model_title = None
    _modeling_task = None
    _library = Iterable[str]
    _version = Iterable[str]
    _jaqpotpy_version = None
    _jaqpotpy_docker = None
    _optimizer = None

    @property
    def smiles(self):
        """Get or set the SMILES representation of molecules."""
        return self._smiles

    @smiles.setter
    def smiles(self, values):
        self._smiles = values

    @property
    def descriptors(self):
        """Get or set the molecular featurizer."""
        return self._descriptors

    @descriptors.setter
    def descriptors(self, value):
        self._descriptors = value

    @property
    def doa(self):
        """Get or set the Domain of Applicability object."""
        return self._doa

    @doa.setter
    def doa(self, value):
        self._doa = value

    @property
    def X(self):
        """Get or set the input features."""
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    @property
    def Y(self):
        """Get or set the output features."""
        return self._Y

    @Y.setter
    def Y(self, value):
        self._Y = value

    @property
    def external_feats(self) -> Iterable[str]:
        """Get or set the external features."""
        return self._external_feats

    @external_feats.setter
    def external_feats(self, value):
        self._external_feats = value

    @property
    def model(self):
        """Get or set the underlying model."""
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def model_title(self):
        """Get or set the title of the model."""
        return self._model_title

    @model_title.setter
    def model_title(self, value):
        self._model_title = value

    @property
    def prediction(self):
        """Get or set the model predictions."""
        return self._prediction

    @prediction.setter
    def prediction(self, value):
        self._prediction = value

    @property
    def probability(self):
        """Get or set the prediction probabilities."""
        return self._probability

    @probability.setter
    def probability(self, value):
        self._probability = value

    @property
    def library(self):
        """Get or set the library used."""
        return self._library

    @library.setter
    def library(self, value):
        self._library = value

    @property
    def optimizer(self):
        """Get or set the optimizer used."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def version(self):
        """Get or set the version of the model."""
        return self._version

    @version.setter
    def version(self, value):
        self._version = value

    @property
    def jaqpotpy_version(self):
        """Get or set the version of jaqpotpy."""
        return self._jaqpotpy_version

    @jaqpotpy_version.setter
    def jaqpotpy_version(self, value):
        self._jaqpotpy_version = value

    @property
    def modeling_task(self):
        """Get or set the description of the modeling task."""
        return self._modeling_task

    @modeling_task.setter
    def modeling_task(self, value):
        self._modeling_task = value

    @property
    def jaqpotpy_docker(self):
        """Get or set the Docker information for jaqpotpy."""
        return self._jaqpotpy_docker

    @jaqpotpy_docker.setter
    def jaqpotpy_docker(self, value):
        self._jaqpotpy_docker = value

    def fit(self):
        """Fit the model to the data."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def predict(self, X):
        """Predict using the model.

        Args:
            X (Any): Input data for prediction.

        Returns:
            Any: Predictions for the input data.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
