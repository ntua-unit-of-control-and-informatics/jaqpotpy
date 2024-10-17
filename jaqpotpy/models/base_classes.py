from jaqpotpy.doa import DOA
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from typing import Any, Iterable


class Model(object):
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
        return self._smiles

    @smiles.setter
    def smiles(self, values):
        self._smiles = values

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, value):
        self._descriptors = value

    @property
    def doa(self):
        return self._doa

    @doa.setter
    def doa(self, value):
        self._doa = value

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, value):
        self._Y = value

    @property
    def external_feats(self) -> Iterable[str]:
        return self._external_feats

    @external_feats.setter
    def external_feats(self, value):
        self._external_feats = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def model_title(self):
        return self._model_title

    @model_title.setter
    def model_title(self, value):
        self._model_title = value

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, value):
        self._prediction = value

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, value):
        self._probability = value

    @property
    def library(self):
        return self._library

    @library.setter
    def library(self, value):
        self._library = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, value):
        self._version = value

    @property
    def jaqpotpy_version(self):
        return self._jaqpotpy_version

    @property
    def modeling_task(self):
        return self._modeling_task

    @modeling_task.setter
    def modeling_task(self, value):
        self._modeling_task = value

    @jaqpotpy_version.setter
    def jaqpotpy_version(self, value):
        self._jaqpotpy_version = value

    @property
    def jaqpotpy_docker(self):
        return self._jaqpotpy_docker

    @jaqpotpy_docker.setter
    def jaqpotpy_docker(self, value):
        self._jaqpotpy_docker = value

    def fit(self):
        raise NotImplementedError("Not implemented")

    def predict(self, X):
        raise NotImplementedError("Not implemented")
