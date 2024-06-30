"""
Author: Ioannis Pitoskas (jpitoskas@gmail.com)
"""

from . import MulticlassModelTrainer

import torch
import torch_geometric
import sklearn

import io
import base64
import pickle
from sklearn.exceptions import NotFittedError
from jaqpotpy.schemas import Feature
from typing import Optional
import inspect


class MulticlassGraphModelWithExternalTrainer(MulticlassModelTrainer):    
    """
    Trainer class for Multiclass Classification using both Graph and Fully Connected Neural Networks for SMILES and as well external features.
    """

    MODEL_TYPE = 'multiclass-graph-model-with-external'
    """'multiclass-graph-model-with-external'"""

    @classmethod
    def get_model_type(cls):
        return cls.MODEL_TYPE
    
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
            ):
        """
        The MulticlassGraphModelWithExternalTrainer constructor.

        Args:
            model (torch.nn.Module): The torch model to be trained.
            n_epochs (int): Number of training epochs.
            optimizer (torch.optim.Optimizer): The optimizer used for training the model.
            loss_fn (torch.nn.Module): The loss function used for training.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The scheduler used for adjusting the learning rate during training. Default is None.
            device (str, optional): The device on which to train the model. Default is 'cpu'.
            use_tqdm (bool, optional): Whether to use tqdm for progress bars. Default is True.
            log_enabled (bool, optional): Whether logging is enabled. Default is True.
            log_filepath (str or None, optional): Path to the log file. If None, logging is not saved to a file. Default is None.
        
        Example:
        ```
        >>> import torch
        >>> from jaqpotpy.jaqpotpy_torch.models import GraphAttentionNetworkWithExternal
        >>> from jaqpotpy.jaqpotpy_torch.trainers import MulticlassGraphModelWithExternalTrainer
        >>> 
        >>> num_classes = 10
        >>> model = GraphAttentionNetworkWithExternal(graph_input_dim=10, 
        ...                                           num_external_features=8, 
        ...                                           fc_hidden_dims=[16, 32], 
        ...                                           graph_hidden_dims=[64, 64],
        ...                                           graph_output_dim=10,
        ...                                           output_dim=num_classes)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        >>> loss_fn = torch.nn.CrossEntropyLoss()
        >>>
        >>> trainer = MulticlassGraphModelWithExternalTrainer(model, n_epochs=50, optimizer=optimizer, loss_fn=loss_fn)
        ```
        """
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
            )
        
    def get_model_kwargs(self, data):
        """
        Fetch the model's keyword arguments.

        Args:
            data (torch_geometric.data.Data): Data object returned as returned by the Dataloader

        Returns:
            dict: The required model kwargs. Set of keywords: {'*x*', '*edge_index*', '*batch*', '*external*', '*edge_attr*'}. Note that '*edge_attr*' is only present if the model supports edge features.
        """

        kwargs = {}

        kwargs['x'] = data.x
        kwargs['edge_index'] = data.edge_index
        kwargs['batch'] = data.batch
        kwargs['external'] = data.external

        if 'edge_attr' in inspect.signature(self.model.forward).parameters:
            kwargs['edge_attr'] = data.edge_attr

        return kwargs
    

    def prepare_for_deployment(self,
                               featurizer,
                               external_preprocessor,
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
            featurizer (object): The featurizer used to transform the SMILES to graph representations before training the model.
            external_preprocessor (object): The preprocessor used to transform the external data before training the model.
            endpoint_name (str): The name of the endpoint for the deployed model.
            name (str): The name to be assigned to the deployed model.
            description (str, optional): A description for the model to be deployed. Default is None.
            visibility (str, optional): Visibility of the deployed model. Can be 'PUBLIC', 'PRIVATE' or 'ORG_SHARED'. Default is 'PUBLIC'.
            reliability (int, optional): The models reliability. Default is None.
            pretrained (bool, optional): Indicates if the model is pretrained. Default is False.
            meta (dict, optional): Additional metadata for the model. Default is an empty dictionary.
        
        
        Returns:
            dict: The data to be sent to the API of Jaqpot in JSON format.
                  Note that in this case, the '*additional_model_params*' key contains a nested dictionary with they keys: {'*featurizer*', '*external_preprocessor*'}.
        """
        
        if not external_preprocessor._sklearn_is_fitted:
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})
        
        if len(external_preprocessor.new_columns_) != self.model.num_external_features:
            raise ValueError(f"Size {len(external_preprocessor.new_columns_)} of 'external_preprocessor.new_columns_' must match the number of {self.model.num_external_features} external features that the model expects as input.")

        self.model = self.model.cpu()
        model_scripted = torch.jit.script(self.model)
        model_buffer = io.BytesIO()
        torch.jit.save(model_scripted, model_buffer)
        model_buffer.seek(0)
        model_scripted_base64 = base64.b64encode(model_buffer.getvalue()).decode('utf-8')

        featurizer_buffer = io.BytesIO()
        pickle.dump(featurizer, featurizer_buffer)
        featurizer_buffer.seek(0)
        featurizer_pickle_base64 = base64.b64encode(featurizer_buffer.getvalue()).decode('utf-8')
        
        external_preprocessor_buffer = io.BytesIO()
        pickle.dump(external_preprocessor, external_preprocessor_buffer)
        external_preprocessor_buffer.seek(0)
        external_preprocessor_pickle_base64 = base64.b64encode(external_preprocessor_buffer.getvalue()).decode('utf-8')
        

        additional_model_params = {
            'featurizer': featurizer_pickle_base64,
            'external_preprocessor': external_preprocessor_pickle_base64
        }

        smiles_feature = Feature(
            name='SMILES',
            featureDependency='INDEPENDENT',
            possibleValues=[],
            featureType='SMILES',
        )

        external_feature_names = external_preprocessor.new_columns_
        external_features = []

        for item in Feature.get_feature_names_and_possible_values_from_column_names(external_feature_names):
            feature_name = item['name']
            possibleValues = item['possibleValues']

            feature = Feature(
                name=feature_name,
                featureDependency='INDEPENDENT',
                possibleValues=possibleValues,
                featureType='CATEGORICAL' if possibleValues != [] else 'FLOAT'
            )
            
            external_features.append(feature)
        
        independentFeatures = [smiles_feature] + external_features

        num_classes = self.model.output_dim
        dependentFeatures = [
            Feature(name=endpoint_name, featureDependency='DEPENDENT', possibleValues=[str(i) for i in range(num_classes)], featureType='CATEGORICAL')
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
