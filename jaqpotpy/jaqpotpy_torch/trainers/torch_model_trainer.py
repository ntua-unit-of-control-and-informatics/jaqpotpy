from abc import ABC, abstractmethod
import torch
import logging
import sys

import jaqpotpy
import requests
import os

class TorchModelTrainer(ABC):
    @property
    @abstractmethod
    def model_type(self):
        pass

    def __init__(self,
                 model,
                 n_epochs,
                 optimizer,
                 loss_fn,
                 device='cpu',
                 use_tqdm=True,
                 log_enabled=True,
                 log_filepath=None):
        """
        Args:
            model (torch.nn.Module): The torch model to be trained.
            n_epochs (int): Number of training epochs.
            optimizer (torch.optim.Optimizer): The optimizer used for training the model.
            loss_fn (torch.nn.Module): The loss function used for training.
            device (str, optional): The device on which to train the model. Default is 'cpu'.
            use_tqdm (bool, optional): Whether to use tqdm for progress bars. Default is True.
            log_enabled (bool, optional): Whether logging is enabled. Default is True.
            log_filepath (str or None, optional): Path to the log file. If None, logging is not saved to a file. Default is None.
        """
        self.model = model
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.use_tqdm = use_tqdm
        self.current_epoch = 0
        self.log_enabled = log_enabled
        self.log_filepath = os.path.relpath(log_filepath)
        self.json_data_for_deployment = None

        self.logger = self._setup_logger()

        # print(vars(self))
        # exit()

    # def check_reach_epoch_limit(self, func):
    #     def wrapper(self, *args, **kwargs):
    #         if self.current_epoch >= self.n_epochs:
    #             self.logger.info(f"WARNING: Model has already been trained for the indicated number epochs but will train for {self.n_epochs} more.")
            
    #         return func(self, *args, **kwargs)
    #     return wrapper
    
    def _setup_logger(self):

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
        handlers=[logging.StreamHandler(sys.stdout)]
        if self.log_filepath:
            if os.path.exists(self.log_filepath):
                raise ValueError(f"File already exists: {self.log_filepath}")
            log_file_handler = logging.FileHandler(self.log_filepath)
            handlers.append(log_file_handler)

        log_format = '%(message)s'
        formatter = logging.Formatter(log_format)
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    # @check_reach_epoch_limit
    @abstractmethod
    def train(self, train_loader, val_loader=None):
        """
        Train the model.

        Args:
            train_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the training dataset.            
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader], optional): DataLoader for the validation dataset.
        Returns:
            None
        """
        pass

    @abstractmethod
    def evaluate(self, val_loader):
        """
        Evaluate the model.

        Args:
            loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the evaluation dataset.
        """
        pass
    
    @abstractmethod
    def prepare_for_deployment(self, *args, **kwargs):
        """
        Prepare the model for deployment on Jaqpot.
        """
        pass

    @classmethod
    def _model_data_as_json(cls,
                            actualModel,
                            model_type,
                            featurizer,
                            public,
                            libraries,
                            independentFeatures,
                            dependentFeatures,
                            additional_model_params,
                            reliability=0,
                            pretrained=False,
                            meta=dict()
                            ):
        
        data = {
            'meta': meta,
            'public': public,
            'type': model_type,
            'jaqpotpyVersion': str(jaqpotpy.__version__),
            'libraries' : libraries,
            'independentFeatures': independentFeatures,
            'dependentFeatures': dependentFeatures,
            'additional_model_params': additional_model_params,
            'reliability': reliability,
            'pretrained': pretrained,
            'actualModel': actualModel,
            'featurizer': featurizer,
        }

        return data

    
    def deploy_model(self, token):

        if self.json_data_for_deployment is None:
            no_json_err_msg = f"No JSON data to deploy. Note that prepare_for_deployment() must be called first to prepare the JSON to be deployed. Please check the function documentation for the specific arguments needed for the current class {self.__class__.__name__}."
            raise ValueError(no_json_err_msg)

        headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
        
        url = "http://localhost:8006/api/v1/models/upload/"
        response = requests.post(url, json=self.json_data_for_deployment, headers=headers)

        return response
    
    



    
