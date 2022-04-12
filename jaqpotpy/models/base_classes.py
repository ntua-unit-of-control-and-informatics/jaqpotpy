from jaqpotpy.doa.doa import DOA
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from typing import Any, Iterable
import pandas as pd
import pickle
import numpy as np
import base64
import torch.nn.functional as nnf
# from jaqpotpy import Jaqpot


class Model(object):
    _model: Any
    _doa: DOA
    _descriptors: MolecularFeaturizer
    _preprocessors: []
    _preprocessor_names: []
    _preprocessor_y_names: []
    _preprocessors_y: []
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
    def external(self):
        return self._external

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
    def preprocessing_y(self):
        return self._preprocessors_y

    @preprocessing_y.setter
    def preprocessing_y(self, value):
        self._preprocessors_y = value

    @property
    def preprocessor_names(self):
        return self._preprocessor_names

    @preprocessor_names.setter
    def preprocessor_names(self, value):
        self._preprocessor_names = value

    @property
    def preprocessor_y_names(self):
        return self._preprocessor_names

    @preprocessor_y_names.setter
    def preprocessor_y_names(self, value):
        self._preprocessor_y_names = value

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
    def external(self) -> dict:
        return self._external

    @external.setter
    def external(self, value):
        self._external = value

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

    def __train__(self):
        raise NotImplemented("Not implemented")

    def __eval__(self):
        raise NotImplemented("Not implemented")

    def infer(self):
        raise NotImplemented("Not implemented")


class MolecularModel(Model):

    def __call__(self, smiles, external=None):
        self._smiles = smiles
        self._external = external
        self.infer()

    def save(self):
        if self._model_title:
            with open(self._model_title + ".jmodel", 'wb') as f:
                pickle.dump(self, f)
        else:
            with open("jaqpot_model" + ".jmodel", 'wb') as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def deploy_on_jaqpot(self, jaqpot, description, model_title: str = None):
        jaqpot.deploy_jaqpotpy_molecular_model(self, description=description, title=model_title)

    @classmethod
    def load_from_jaqpot(cls, jaqpot, id: str):
        model = jaqpot.get_raw_model_by_id(id)
        raw_model = base64.b64decode(model['actualModel'][0])
        model: MolecularModel = pickle.loads(raw_model)
        return model

    def infer(self):
        self._prediction = []
        self._probability = []
        if self._smiles:
            data = self._descriptors.featurize_dataframe(self._smiles)
            if self.external:
                ext = pd.DataFrame.from_dict(self.external)
                data = pd.concat([data, ext], axis=1)
            graph_data_list = []
            if self._descriptors.__name__ == 'MolGraphConvFeaturizer':
                graph_data = data['MoleculeGraph']
                for g in data['MoleculeGraph'].to_list():
                    import torch
                    from torch_geometric.data import Data
                    dat = Data(x=torch.FloatTensor(g.node_features)
                               , edge_index=torch.LongTensor(g.edge_index)
                               , edge_attr=g.edge_features
                               , num_nodes=g.num_nodes)
                    graph_data_list.append(dat)
                self._prediction = []
            if self._X != ['TorchMolGraph'] and self.X != ['OneHotSequence'] and self.X != ["SmilesImage"]:
                data = data[self._X].to_numpy()
            elif self._X == ['SmilesImage']:
                datas = []
                for v in data.values:
                    datas.append(v[0].transpose(2, 0, 1))
                data = np.array(datas)
            elif self.X == ['OneHotSequence']:
                data_list = []
                for d in data.values:
                    data_list.append(d[0])
                data = np.array(data_list)
            if self.doa:
                if self.doa.__name__ == 'SmilesLeverage':
                    self.doa.predict(self._smiles)
                else:
                    self.doa.predict(data)
            try:
                if self.preprocessing:
                    for f in self.preprocessing:
                        data = f.transform(data)
            except AttributeError as e:
                pass
            if self.library == ['sklearn']:
                preds = self.model.predict(data)
                for p in preds:
                    try:
                        if self.preprocessing_y:
                            for f in self.preprocessing_y:
                                p = f.inverse_transform(p.reshape(1, -1))
                    except AttributeError as e:
                        pass
                    self._prediction.append(p.tolist())
                try:
                    probs = self.model.predict_proba(data)
                    for prob in probs:
                        self._probability.append(prob)
                except AttributeError as e:
                    pass
            if self.library == ['torch_geometric', 'torch']:
                self.model.eval()
                # self.model.no_grad()
                # torch.no_grad()
                from torch_geometric.loader import DataLoader
                data_loader = DataLoader(graph_data_list, batch_size=len(graph_data_list))
                for g in data_loader:
                    pred = self.model(g)
                    if self.modeling_task == "classification":
                        for p in pred:
                            prob = nnf.softmax(p, dim=0)
                            self._probability.append(prob.detach().numpy().tolist())
                            # self._probability.append(p.detach().numpy().tolist())
                        pred = pred.argmax(dim=1)
                        preds = pred.detach().numpy()
                        for p in preds:
                            try:
                                if self.preprocessing_y:
                                    for f in self.preprocessing_y:
                                        p = f.inverse_transform(p)
                            except AttributeError as e:
                                pass
                            self._prediction.append(p.tolist())
                    else:
                        preds = pred.detach().numpy()
                        for p in preds:
                            self._prediction.append(p.tolist())
            if self.library == ['torch']:
                self.model.eval()
                from torch.utils.data import DataLoader
                data_loader = DataLoader(data, batch_size=len(data))
                for g in data_loader:
                    pred = self.model(g.float())
                    if self.modeling_task == "classification":
                        for p in pred:
                            prob = nnf.softmax(p, dim=0)
                            self._probability.append(prob.detach().numpy().tolist())
                            # self._probability.append(p.detach().numpy().tolist())
                        pred = pred.argmax(dim=1)
                        preds = pred.detach().numpy()
                        for p in preds:
                            try:
                                if self.preprocessing_y:
                                    for f in self.preprocessing_y:
                                        p = f.inverse_transform(p)
                            except AttributeError as e:
                                pass
                            self._prediction.append([p.tolist()])
                    else:
                        preds = pred.detach().numpy()
                        for p in preds:
                            try:
                                if self.preprocessing_y:
                                    for f in self.preprocessing_y:
                                        p = f.inverse_transform(p)
                            except AttributeError as e:
                                pass
                            self._prediction.append([p.tolist()])
            # else:
            #     preds = self.model.predict(data)
            # for p in preds:
            #     self._prediction.append([p[0]])
            return self
        else:
            pass