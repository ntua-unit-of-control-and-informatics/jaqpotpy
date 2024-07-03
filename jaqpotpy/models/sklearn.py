import sklearn.pipeline
from jaqpotpy.models.base_classes import Model
from jaqpotpy.doa.doa import DOA
from typing import Any, Union, Dict, Optional
from jaqpotpy.datasets.molecular_datasets import JaqpotpyDataset
from jaqpotpy.datasets.material_datasets import CompositionDataset, StructureDataset
from jaqpotpy.models import Evaluator, Preprocess, MolecularModel, MaterialModel
import sklearn
from jaqpotpy.cfg import config
import jaqpotpy
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime import InferenceSession
import numpy as np


class SklearnModel(Model):

    def __init__(self, dataset: JaqpotpyDataset, doa: DOA, model: Any,
                  preprocessor: Preprocess = None,  evaluator: Evaluator = None):
        self.x_cols = dataset.x_cols
        self.dataset = dataset
        self.descriptors = dataset.featurizer
        self.model = model
        self.pipeline = None
        self.pipeline_y = None
        self.pipeline_y_onnx = None
        self.trained_model = None
        self.doa = doa
        self.trained_doa = None
        self.evaluator = evaluator
        self.preprocess = preprocessor
        self.preprocessing_y = None
        self.preprocessor_y_names = None
        self.library = ['sklearn']
        self.version = [sklearn.__version__]
        self.jaqpotpy_version = jaqpotpy.__version__
        self.jaqpotpy_docker = config.jaqpotpy_docker
        self.task = self.dataset.task
        self.onnx_model = None
        self.onnx_opset = None

    def fit(self):
        #Get X and y from dataset
        X = self.dataset.__get_X__()
        y = self.dataset.__get_Y__()

        if self.doa:
            self.trained_doa = self.doa.fit(X=X)
        
        #if preprocessing was applied to either X,y or both
        if self.preprocess is not None:
            # Apply preprocessing on design matrix X
            pre_keys = self.preprocess.classes.keys()
            if len(pre_keys) > 0:
                pipeline = sklearn.pipeline.Pipeline(steps = list(self.preprocess.classes.items()))
            pipeline.steps.append(('model', self.model))
            self.pipeline = pipeline

            # Apply preprocessing of response vector y
            pre_y_keys = self.preprocess.classes_y.keys()
            if len(pre_y_keys) > 0:
                if self.task == "classification":
                    raise ValueError("Classification levels cannot be preprocessed")
                else:
                    preprocess_names_y = []
                    preprocess_classes_y = []
                    for pre_y_key in pre_y_keys:
                        pre_y_function = self.preprocess.classes_y.get(pre_y_key)
                        y_scaled = pre_y_function.fit_transform(y).ravel()
                        self.preprocess.register_fitted_class_y(pre_y_key, pre_y_function)
                        preprocess_names_y.append(pre_y_key)
                        preprocess_classes_y.append(pre_y_function)
                    self.preprocessing_y = preprocess_classes_y
                    self.preprocessor_y_names = preprocess_names_y

                    self.trained_model = self.pipeline.fit(X.to_numpy(), y_scaled)
            else:
                self.trained_model = self.pipeline.fit(X.to_numpy(), y.to_numpy().ravel())   
        #case where no preprocessing was provided
        else:
            self.trained_model = self.model.fit(X.to_numpy(), y.to_numpy().ravel())
        
        name = self.model.__class__.__name__ + "_ONNX"
        self.onnx_model = convert_sklearn(self.trained_model, initial_types=[('float_input', FloatTensorType([None, X.to_numpy().shape[1]]))], name=name)
        self.onnx_opset = self.onnx_model.opset_import[0].version

        if self.evaluator:
            self.__eval__()
        return self

    def predict(self,  dataset: JaqpotpyDataset):
        sklearn_prediction = self.trained_model.predict(dataset.X.to_numpy())
        if self.preprocess is not None:
            if self.preprocessing_y:
                for f in self.preprocessing_y:
                    sklearn_prediction = f.inverse_transform(sklearn_prediction.reshape(1, -1))[0].flatten()

        return sklearn_prediction
    
    def predict_onnx(self,  dataset: JaqpotpyDataset):
        sess = InferenceSession(self.onnx_model.SerializeToString())
        onnx_prediction = sess.run(None, {"float_input": dataset.X.to_numpy().astype(np.float32)})
        if self.preprocess is not None:
            if self.preprocessing_y:
                for f in self.preprocessing_y:
                    onnx_prediction[0] = f.inverse_transform(onnx_prediction[0])
        return onnx_prediction[0].flatten()

    def __eval__(self):
        if self.evaluator.dataset.df is not None:
            pass
        else:
            self.evaluator.dataset.create()
        X = self.evaluator.dataset.__get_X__()
        if self.preprocess:
            pre_keys = self.preprocess.classes.keys()
            for pre_key in pre_keys:
                preprocess_func = self.preprocess.fitted_classes.get(pre_key)
                X = preprocess_func.transform(X)
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
                 , eval: Evaluator = None, preprocess: Preprocess = None):
        # super(InMemMolModel, self).__init__(dataset=dataset, doa=doa, model=model)
        self.dataset = dataset
        self.model = model
        self.doa = doa
        self.doa_fitted = None
        self.external = None
        self.evaluator: Evaluator = eval
        self.preprocess: Preprocess = preprocess
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
            self.doa_fitted = self.doa.fit(X=self.dataset.__get_X__())

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

    def predict(self, X):
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