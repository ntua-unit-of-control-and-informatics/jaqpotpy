"""
Author: Ioannis Pitoskas
Contact: jpitoskas@gmail.com
"""

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
    """
    Trainer class for Regression using a Fully Connected Network on tabular data.
    
    Args:
        model (torch.nn.Module): The torch model to be trained.
        n_epochs (int): Number of training epochs.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        loss_fn (torch.nn.Module): The loss function used for training.
        device (str, optional): The device on which to train the model. Default is 'cpu'.
        use_tqdm (bool, optional): Whether to use tqdm for progress bars. Default is True.
        log_enabled (bool, optional): Whether logging is enabled. Default is True.
        log_filepath (str or None, optional): Path to the log file. If None, logging is not saved to a file. Default is None.
        normalization_mean (float): Mean used to normalize the true values of the regression variables before model training. 
        normalization_std' (float): Standard deviation used to normalize the true values of the regression variables before model training.    
    """

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
            scheduler=None, 
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
            scheduler=scheduler,
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
                               endpoint_name: str,
                               name: str,
                               description: Optional[str] = None,
                               visibility: str = 'PUBLIC',
                               reliability: Optional[int] = None,
                               pretrained: bool = False,
                               meta: dict = dict()
                               ):
        """
        Prepare the model for deployment on Jaqpot.
        
        Args:
            preprocessor (object): The preprocessor used to transform the tabular data before training the model.
            endpoint_name (str): The name of the endpoint for the deployed model.
            name (str): The name to be assigned to the deployed model.
            description (str, optional): A description for the model to be deployed. Default is None.
            visibility (str, optional): Visibility of the deployed model. Can be 'PUBLIC', 'PRIVATE' or 'ORG_SHARED'. Default is 'PUBLIC'.
            reliability (int, optional): The models reliability. Default is None.
            pretrained (bool, optional): Indicates if the model is pretrained. Default is False.
            meta (dict, optional): Additional metadata for the model. Default is an empty dictionary.
        """
        
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
   