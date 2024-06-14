from . import RegressionModelTrainer

import torch
import sklearn

import io
import base64
import pickle
from sklearn.exceptions import NotFittedError
from jaqpotpy.schemas import Feature
from typing import Optional


class RegressionFCModelTrainer(RegressionModelTrainer):
    model_type = 'regression-fc-model'

    @classmethod
    def get_model_type(cls):
        return cls.model_type
    
    def __init__(
            self, 
            model, 
            n_epochs, 
            optimizer, 
            loss_fn, 
            device='cpu', 
            use_tqdm=True,
            log_enabled=True,
            log_filepath=None,
            normalization_mean=0.5,
            normalization_std=1.0
            ):
        
        super().__init__(
            model=model,
            n_epochs=n_epochs,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            use_tqdm=use_tqdm,
            log_enabled=log_enabled,
            log_filepath=log_filepath,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std
            )
    
    def get_model_kwargs(self, data):

        kwargs = {}

        X, _ = data
        kwargs['x'] = X

        return kwargs
    
    def prepare_for_deployment(self,
                               preprocessor,
                               endpoint_name,
                               name: str,
                               description: Optional[str] = None,
                               visibility='PUBLIC',
                               reliability=True,
                               pretrained=False,
                               meta=dict()
                               ):
        
        if not preprocessor._sklearn_is_fitted:
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})
        
        if len(preprocessor.new_columns_) != self.model.input_dim:
            raise ValueError(f"Size {len(preprocessor.new_columns_)} of 'preprocessor.new_columns_' must match the number of {self.model.num_features} features that the model expects as input.")


        model_scripted = torch.jit.script(self.model)
        model_buffer = io.BytesIO()
        torch.jit.save(model_scripted, model_buffer)
        model_buffer.seek(0)
        model_scripted_base64 = base64.b64encode(model_buffer.getvalue()).decode('utf-8')

        preprocessor_buffer = io.BytesIO()
        pickle.dump(preprocessor, preprocessor_buffer)
        preprocessor_buffer.seek(0)
        preprocessor_pickle_base64 = base64.b64encode(preprocessor_buffer.getvalue()).decode('utf-8')
        
        additional_model_params = {
            'normalization_mean': self.normalization_mean,
            'normalization_std': self.normalization_std,
            'preprocessor': preprocessor_pickle_base64
        }

        feature_names = preprocessor.new_columns_
        features = []

        for item in Feature.get_feature_names_and_possible_values_from_column_names(feature_names):
            feature_name = item['name']
            possibleValues = item['possibleValues']

            feature = Feature(
                name=feature_name,
                featureDependency='INDEPENDENT',
                possibleValues=possibleValues,
                featureType='CATEGORICAL' if possibleValues != [] else 'NUMERICAL'
            )
            
            features.append(feature)
        
        independentFeatures = features

        dependentFeatures = [
            Feature(name=endpoint_name, featureDependency='DEPENDENT', possibleValues=[], featureType='NUMERICAL')
        ]


        self.json_data_for_deployment = self._model_data_as_json(actualModel=model_scripted_base64,
                                                                 name=name,
                                                                 description=description,
                                                                 model_type=self.get_model_type(),
                                                                 visibility=visibility,
                                                                 independentFeatures=independentFeatures,
                                                                 dependentFeatures=dependentFeatures,
                                                                 additional_model_params=additional_model_params,
                                                                 reliability=reliability,
                                                                 pretrained=pretrained,
                                                                 meta=meta
                                                                 )
        
        return self.json_data_for_deployment
   