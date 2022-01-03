from jaqpotpy.datasets.dataset_base import MolecularTabularDataset
from typing import Any
from jaqpotpy.doa.doa import DOA
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from typing import Any, Iterable


class Model(object):
    # def __init__(self, dataset: MolecularTabularDataset, doa: DOA, model: Any):
    #     self.dataset = dataset
    #     self.DOA = doa
    #     self.model = model

    # @property
    # def dataset(self):
    #     return self.dataset
    #
    # @dataset.setter
    # def dataset(self, value):
    #     self._dataset = value
    #
    # @property
    # def DOA(self):
    #     return self.DOA
    #
    # @DOA.setter
    # def DOA(self, value):
    #     self._DOA = value
    #
    # @property
    # def model(self):
    #     return self.model
    #
    # @model.setter
    # def model(self, value):
    #     self._model = value

    def __train__(self):
        raise NotImplemented("Not implemented")

    def __eval__(self):
        raise NotImplemented("Not implemented")


class InMemMolModel(Model):

    def __init__(self, dataset: MolecularTabularDataset, doa: DOA, model: Any, eval:Any):
        # super(InMemMolModel, self).__init__(dataset=dataset, doa=doa, model=model)
        self.dataset = dataset
        self.model = model
        self.doa = doa
        self.doa_m = None
        # self.trained_model = None

    def __call__(self, *args, **kwargs):
        self

    def __train__(self):
        self.dataset.create()
        self.doa_m = self.doa.fit(X=self.dataset.__get_X__())
        self.trained_model = self.model.fit(self.dataset.__get_X__(), self.dataset.__get_Y__())
        model = MolecularModel()
        model.descriptors = self.dataset.featurizer
        model.doa = self.doa
        model.model = self.trained_model
        model.preprocessing = []
        model.X = self.dataset.X
        return model

    def __predict__(self, X):
        # self.dataset.
        data = self.dataset.featurizer.featurize_dataframe(X)
        data = data[self.dataset.X].to_numpy()
        return self.model.predict(data)

    def __eval__(self):
        raise NotImplementedError


class MolecularModel(Model):
    _model: Any
    _doa: DOA
    _descriptors: Iterable[MolecularFeaturizer]
    _preprocessors: Iterable[Any]
    _X: Iterable[str]
    _X_indices: Iterable[int]

    # def __init__(self):
        # self._smiles: str = smiles

    def __call__(self, smiles):
        self._smiles = smiles
        self.infer()

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
    def preprocessing(self):
        return self._preprocessors

    @preprocessing.setter
    def preprocessing(self, value):
        self._preprocessors = value

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def infer(self):
        if self._smiles:
            data = self._descriptors.featurize_dataframe(self._smiles)
            data = data[self._X].to_numpy()
            predict = self.model.predict(data)
            return predict
        else:
            pass
