from jaqpotpy.models.base_classes import Model
from jaqpotpy.doa.doa import DOA
from typing import Any, Union, Dict, Optional
from jaqpotpy.datasets import MolecularDataset
from jaqpotpy.datasets.material_datasets import CompositionDataset, StructureDataset
from jaqpotpy.models import Evaluator, Preprocesses, MolecularModel, MaterialModel
import sklearn
from jaqpotpy.cfg import config
import jaqpotpy
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import numpy as np


class MolecularSKLearn(Model):

    def __init__(self, dataset: MolecularDataset, doa: DOA, model: Any
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
        self._initial_types = None

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
                self.doa_m = self.doa.fit(self.dataset.smiles)
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

        X = X.astype(float).copy()
        self.trained_model = self.model.fit(X, y)
        if type(self.dataset.featurizer).__name__ == "RDKitDescriptors":
            model.descriptors = "RDKitDescriptors"
        elif type(self.dataset.featurizer).__name__ == "MACCSKeysFingerprint":
            model.descriptors = "MACCSKeysFingerprint"
        else:
            model.descriptors = self.dataset.featurizer
        model.doa = self.doa
        model.model = self.trained_model
        model.X = self.dataset.X
        model.Y = self.dataset.y
        model.library = ['sklearn']
        model.version = [sklearn.__version__]
        model.jaqpotpy_version = jaqpotpy.__version__
        model.jaqpotpy_docker = config.jaqpotpy_docker
        model.external_feats = self.dataset.external
        model.inference_model = self.__convert_to_onnx__(X.shape[1])
        self.inference_model = model.inference_model
        if self.evaluator:
            self.__eval__()
        return model

    def __convert_to_onnx__(self, num_columns):

        name = self.model.__class__.__name__ + "_ONNX"
        initial_types = [('float_input',FloatTensorType([None, num_columns]))]
        res = convert_sklearn(self.model, initial_types=initial_types, name=name, target_opset=None,
                              options=None, black_op=None, white_op=None, final_types=None,
                              verbose=0)
        return res

    def __predict__(self, X):
        pre_keys = self.preprocess.fitted_classes.keys()
        for pre_key in pre_keys:
            pre_function = self.preprocess.fitted_classes.get(pre_key)
            X = pre_function.transform(X)
        data = self.dataset.featurizer.featurize(X)
        sess = rt.InferenceSession(self.model.inference_model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        pred_onx = sess.run(None, {input_name: data.astype(np.float32)})
        return pred_onx[0].flatten()

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
        sess = rt.InferenceSession(self.inference_model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        X = np.array(X.astype(float).copy())
        preds = sess.run(None, {input_name: X.astype(np.float32)})
        preds = preds[0].flatten()
        # preds = self.trained_model.predict(X)
        preds_t = []
        for p in preds:
            try:
                if self.preprocessing_y:
                    for f in self.preprocessing_y:
                        p = f.inverse_transform(p.reshape(1, -1))
                        preds_t.append(p)
            except AttributeError as e:
                pass
            preds_t.append(p)
        eval_keys = self.evaluator.functions.keys()
        for eval_key in eval_keys:
            eval_function = self.evaluator.functions.get(eval_key)
            try:
                if self.preprocessing_y:
                    for f in self.preprocessing_y:
                        truth = f.inverse_transform(self.evaluator.dataset.__get_Y__())
                    print(eval_key + ": " + str(eval_function(truth, preds_t)))
            except AttributeError as e:
                print(eval_key + ": " + str(eval_function(self.evaluator.dataset.__get_Y__(), preds_t)))
                pass
            # print(eval_key + ": " + str(eval_function(self.evaluator.dataset.__get_Y__(), preds_t)))
            #print(eval_key + ": " + str(eval_function(self.evaluator.dataset.__get_Y__(), preds)))



class MaterialSKLearn(Model):

    def __init__(self, dataset: Union[StructureDataset, CompositionDataset], doa: DOA, model: Any, fillna: Optional[Union[Dict, float, int]] = None
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
        self.fillna = fillna
        # self.trained_model = None

    def __call__(self, smiles):
        self

    def fit(self):
        model = MaterialModel()
        if self.dataset.df is not None:
            pass
        else:
            self.dataset.create()

        self.dataset.df.fillna(self.fillna, inplace=True)

        if self.doa:
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
        model.descriptors = self.dataset.featurizer
        model.doa = self.doa
        model.model = self.trained_model
        model.X = self.dataset.X
        model.Y = self.dataset.y
        model.library = ['sklearn']
        model.version = [sklearn.__version__]
        model.jaqpotpy_version = jaqpotpy.__version__
        model.external_feats = self.dataset.external
        if self.evaluator:
            self.__eval__()
        return model

    def __predict__(self, X):
        pre_keys = self.preprocess.fitted_classes.keys()
        for pre_key in pre_keys:
            pre_function = self.preprocess.fitted_classes.get(pre_key)
            X = pre_function.transform(X)
        data = self.dataset.featurizer.featurize_dataframe(X)
        data.df.fillna(self.fillna, inplace=True)
        data = data[self.dataset.X].to_numpy()
        return self.model.predict(data)

    def __eval__(self):
        if self.evaluator.dataset.df is not None:
            pass
        else:
            self.evaluator.dataset.create()
        self.evaluator.dataset.df.fillna(self.fillna, inplace=True)
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