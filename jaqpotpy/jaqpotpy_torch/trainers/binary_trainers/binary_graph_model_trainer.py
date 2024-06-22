"""
Author: Ioannis Pitoskas
Contact: jpitoskas@gmail.com
"""

from . import BinaryModelTrainer

import torch
import torch_geometric
import io
import base64
import pickle
from jaqpotpy.schemas import Feature
from typing import Optional
import inspect


class BinaryGraphModelTrainer(BinaryModelTrainer):
    model_type = 'binary-graph-model'

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
            decision_threshold=0.5,
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
            decision_threshold=decision_threshold
            )
        
    
    def get_model_kwargs(self, data):

        kwargs = {}

        kwargs['x'] = data.x
        kwargs['edge_index'] = data.edge_index
        kwargs['batch'] = data.batch

        if 'edge_attr' in inspect.signature(self.model.forward).parameters:
            kwargs['edge_attr'] = data.edge_attr

        return kwargs
    

    def prepare_for_deployment(self,
                               featurizer,
                               endpoint_name,
                               name: str,
                               description: Optional[str] = None,
                               visibility='PUBLIC',
                               reliability=True,
                               pretrained=False,
                               meta=dict()
                               ):
        
        model_scripted = torch.jit.script(self.model)
        model_buffer = io.BytesIO()
        torch.jit.save(model_scripted, model_buffer)
        model_buffer.seek(0)
        model_scripted_base64 = base64.b64encode(model_buffer.getvalue()).decode('utf-8')


        featurizer_buffer = io.BytesIO()
        pickle.dump(featurizer, featurizer_buffer)
        featurizer_buffer.seek(0)
        featurizer_pickle_base64 = base64.b64encode(featurizer_buffer.getvalue()).decode('utf-8')
        
        additional_model_params = {
            'decision_threshold': self.decision_threshold,
            'featurizer': featurizer_pickle_base64
        }

        independentFeatures = [
            Feature(name='SMILES', featureDependency='INDEPENDENT', possibleValues=[], featureType='SMILES')
        ]

        dependentFeatures = [
            Feature(name=endpoint_name, featureDependency='DEPENDENT', possibleValues=['0', '1'], featureType='CATEGORICAL')
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
