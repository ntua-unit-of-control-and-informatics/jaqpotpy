from jaqpotpy.models.base_classes import Model
from jaqpotpy.doa.doa import DOA
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from typing import Any, Iterable
import pandas as pd
import pickle
from jaqpotpy.datasets.molecular_datasets import MolecularTabularDataset
from jaqpotpy.models import Evaluator, Preprocesses, MolecularModel
import sklearn
from jaqpotpy.cfg import config


class MolecularSKLearn(Model):

    def __init__(self, dataset: MolecularTabularDataset, doa: DOA, model: Any
                 , eval: Evaluator = None, preprocess: Preprocesses = None):
        # super(InMemMolModel, self).__init__(dataset=dataset, doa=doa, model=model)
        self.dataset = dataset
        self.model = model
        self.doa = doa
        self.doa_m = None
        self.external = None
        self.evaluator: Evaluator = eval
        self.preprocess: Preprocesses = preprocess
        self.trained_model = None
        # self.trained_model = None

    def __call__(self, smiles):
        self

    def fit(self):
        model = MolecularModel()
        if self.dataset.df is not None:
            pass
        else:
            self.dataset.create()
        # if not self.dataset.df.empy:
        #     self.dataset.create()
        # else:
        #     pass
        if self.doa:
            if self.doa.__name__ == 'SmilesLeverage':
                self.doa_m = self.doa.fit(self.dataset.smiles_strings)
            else:
                self.doa_m = self.doa.fit(X=self.dataset.__get_X__())
        X = self.dataset.__get_X__()
        y = self.dataset.__get_Y__()
        if self.preprocess:
            pre_keys = self.preprocess.classes.keys()
            preprocess_names = []
            preprocess_classes = []
            for pre_key in pre_keys:
                pre_function = self.preprocess.classes.get(pre_key)
                X = pre_function.fit_transform(X)
                self.preprocess.register_fitted_class(pre_key, pre_function)
                preprocess_names.append(pre_key)
                preprocess_classes.append(pre_function)
            model.preprocessing = preprocess_classes
            model.preprocessor_names = preprocess_names

            pre_y_keys = self.preprocess.classes_y.keys()
            preprocess_names_y = []
            preprocess_classes_y = []
            for pre_y_key in pre_y_keys:
                pre_y_function = self.preprocess.classes_y.get(pre_y_key)
                y = pre_y_function.fit_transform(y)
                self.preprocess.register_fitted_class_y(pre_y_key, pre_y_function)
                preprocess_names_y.append(pre_y_key)
                preprocess_classes_y.append(pre_y_function)
            model.preprocessing_y = preprocess_classes_y
            model.preprocessor_y_names = preprocess_names_y

        self.trained_model = self.model.fit(X, y)
        if self.evaluator:
            self.__eval__()
        model.descriptors = self.dataset.featurizer
        model.doa = self.doa
        model.model = self.trained_model
        model.X = self.dataset.X
        model.Y = self.dataset.y
        model.library = ['sklearn']
        model.version = [sklearn.__version__]
        model.jaqpotpy_version = config.version
        model.external_feats = self.dataset.external
        return model

    def __predict__(self, X):
        pre_keys = self.preprocess.fitted_classes.keys()
        for pre_key in pre_keys:
            pre_function = self.preprocess.fitted_classes.get(pre_key)
            X = pre_function.transform(X)
        data = self.dataset.featurizer.featurize_dataframe(X)
        data = data[self.dataset.X].to_numpy()
        return self.model.predict(data)

    def __eval__(self):
        if self.evaluator.dataset.df is not None:
            pass
        else:
            self.evaluator.dataset.create()
        X = self.evaluator.dataset.__get_X__()
        if self.preprocess:
            pre_keys = self.preprocess.classes.keys()
            for pre_key in pre_keys:
                pre_function = self.preprocess.fitted_classes.get(pre_key)
                X = pre_function.transform(X)
        preds = self.trained_model.predict(X)
        eval_keys = self.evaluator.functions.keys()
        for eval_key in eval_keys:
            eval_function = self.evaluator.functions.get(eval_key)
            print(eval_key + ": " + str(eval_function(self.evaluator.dataset.__get_Y__(), preds)))
