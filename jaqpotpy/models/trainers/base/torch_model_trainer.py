from abc import ABC, abstractmethod, ABCMeta
import torch
import logging
import sys

import requests
import os
from typing import List, Optional, Union
from jaqpotpy.schemas import Feature, Library, Organization
from jaqpotpy.utils.installed_packages import get_installed_packages
import inspect
from torch.optim.lr_scheduler import LambdaLR


class TorchModelTrainerMeta(ABCMeta):
    def __new__(cls, name, bases, dct):
        """
        A metaclass for torch model training, ensuring:
        - 'get_model_type' method is defined as a class method
        """
        if "get_model_type" in dct:
            method = dct["get_model_type"]
            if not isinstance(method, classmethod):
                raise TypeError(f"{name}.get_model_type must be a class method")
        return super().__new__(cls, name, bases, dct)


class TorchModelTrainer(ABC, metaclass=TorchModelTrainerMeta):
    """
    An abstract class for training a model and deploying it on Jaqpot.

    Attributes:
        model (torch.nn.Module): The torch model to be trained.
        n_epochs (int): Number of training epochs.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        loss_fn (torch.nn.Module): The loss function used for training.
        scheduler (torch.optim.lr_scheduler.LRScheduler): The scheduler used for adjusting the learning rate during training.
        device (torch.device): The device on which to train the model.
        use_tqdm (bool): Whether to use tqdm for progress bars.
        current_epoch (int): The epoch on which the trainer has currently reached.
        log_enabled (bool): Whether logging is enabled.
        log_filepath (os.path.relpath or None): Relative path to the log file.
        json_data_for_deployment (dict or None): The data to be sent to the API of Jaqpot in JSON format. Note that `prepare_for_deployment` must be called to compute this attribute.
        logger (logging.Logger): The logger object at INFO level used for logging during model training.
    """

    @classmethod
    @abstractmethod
    def get_model_type(cls):
        """
        Return the type of the model as a string.

        Returns:
            str: The model type.
        """
        pass

    def __init__(
        self,
        model: torch.nn.Module,
        n_epochs: int,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = "cpu",
        use_tqdm: bool = True,
        log_enabled: bool = True,
        log_filepath: Optional[str] = None,
    ):
        """
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
        """
        self.model = model
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.scheduler = (
            scheduler
            if scheduler is not None
            else LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
        )
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.use_tqdm = use_tqdm
        self.current_epoch = 0
        self.log_enabled = log_enabled
        self.log_filepath = (
            os.path.relpath(log_filepath) if log_filepath else log_filepath
        )
        self.json_data_for_deployment = None

        self.logger = self._setup_logger()

        self.model.to(self.device)

    def _setup_logger(self):
        """
        Setup the logger for the training process.
        """

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        handlers = [logging.StreamHandler(sys.stdout)]
        if self.log_filepath:
            if os.path.exists(self.log_filepath):
                raise ValueError(f"File already exists: {self.log_filepath}")
            log_file_handler = logging.FileHandler(self.log_filepath)
            handlers.append(log_file_handler)

        log_format = "%(message)s"
        formatter = logging.Formatter(log_format)
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @classmethod
    def collect_subclass_info(cls):
        """
        Collect information about all subclasses of the current class.

        Returns:
            dict: A dictionary containing information about the subclasses (model type, parent class, if is abstract class or not).
        """
        subclass_info = {}

        for subclass in cls.__subclasses__():

            subclass_info[subclass.__name__] = {
                "model_type": subclass.get_model_type(),
                "parent": cls.__name__,
                "is_abstract": inspect.isabstract(subclass),
            }

            subclass_info.update(subclass.collect_subclass_info())

        return subclass_info

    @classmethod
    def get_subclass_model_types(cls):
        """
        Return the list of all the types of models that inherit from TorchModelTrainer and are implemented in the jaqpotpy_torch.trainers submodule.

        Returns:
            list: A list of model types as strings.
        """
        return [
            v["model_type"]
            for v in cls.collect_subclass_info().values()
            if not v["is_abstract"] and v["model_type"] is not None
        ]

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

    # @abstractmethod
    # def prepare_for_deployment(self, *args, **kwargs):
    #     """
    #     Prepare the model data in JSON format for deployment on Jaqpot.
    #     """
    #     pass

    @staticmethod
    # @enforce_types
    def _model_data_as_json(
        actualModel: str,
        name: str,
        description: Union[str, None],
        model_type: str,
        visibility: str,
        independentFeatures: List[Feature],
        dependentFeatures: List[Feature],
        additional_model_params: dict,
        reliability: Optional[int] = None,
        pretrained: int = False,
        meta: Optional[dict] = None,
        organizations: Optional[List[Organization]] = None,
    ) -> dict:
        """
        Create a JSON representation of the model data for deployment.

        Args:
            actualModel (str): The actual model.
            name (str): The name of the model.
            description (Union[str, None]): The description of the model.
            model_type (str): The type of the model.
            visibility (str): The visibility of the model.
            independentFeatures (List[Feature]): List of independent features (inputs).
            dependentFeatures (List[Feature]): List of dependent features (targets).
            additional_model_params (dict): Additional model parameters.
            reliability (Optional[int]): The reliability of the model. Default is 0.
            pretrained (Optional[int]): Whether the model is pretrained. Default is False.
            meta (Optional[dict]): Additional metadata. Default is None.
            organizations (Optional[List[Organization]]): List of organizations. Default is None.

        Returns:
            dict: The JSON representation of the model data.
        """

        if not isinstance(name, str):
            msg = "'name' should be of type str"
            raise ValueError(msg)

        minLength_name = 3
        maxLength_name = 255
        if not minLength_name <= len(name) <= maxLength_name:
            msg = "Model 'name' length must be between 3 and 255 characters"
            raise ValueError(msg)

        description = "" if description is None else description
        if not isinstance(description, str):
            msg = "'description' should be of type str"
            raise ValueError(msg)

        if not isinstance(model_type, str):
            msg = "'model_type' should be of type str"
            raise ValueError(msg)
        if model_type not in TorchModelTrainer.get_subclass_model_types():
            msg = f"Invalid model_type: '{model_type}'. It must be one of {set(TorchModelTrainer.get_subclass_model_types())}."
            raise ValueError(msg)

        visibility_allowed_values = ["PUBLIC", "ORG_SHARED", "PRIVATE"]
        if visibility not in visibility_allowed_values:
            msg = f"Invalid visibility: {visibility}. Must be one of {visibility_allowed_values}"
            raise ValueError(msg)

        if not isinstance(independentFeatures, list):
            msg = "'independentFeatures' should be of type list"
            raise ValueError(msg)
        #      if any(not isinstance(feature, Feature) for feature in independentFeatures):
        #           msg = "'independentFeatures' should must only include elements of type Feature"
        #            raise ValueError(msg)

        if not isinstance(dependentFeatures, list):
            msg = "'dependentFeatures' should be of type list"
            raise ValueError(msg)
        #        if any(not isinstance(feature, Feature) for feature in independentFeatures):
        #            msg = "'dependentFeatures' should must only include elements of type Feature"
        #            raise ValueError(msg)

        if reliability is not None and not isinstance(reliability, int):
            msg = "'reliability' should be of type int"
            raise ValueError(msg)

        if not isinstance(pretrained, bool):
            msg = "'pretrained' should be of type bool"
            raise ValueError(msg)

        if meta is not None and not isinstance(meta, dict):
            msg = "'meta' should be of type dict"
            raise ValueError(msg)

        if organizations is not None and not isinstance(meta, list):
            msg = "'organizations' should be of type list"
            raise ValueError(msg)
        if organizations is not None and any(
            not isinstance(organization, Organization) for organization in organizations
        ):
            msg = (
                "'organizations' should must only include elements of type Organization"
            )
            raise ValueError(msg)

        if set(feature.name for feature in independentFeatures) & set(
            feature.name for feature in dependentFeatures
        ):
            msg = "There are input and output variables that might have the same naming"
            raise ValueError(msg)

        independent_feature_names = set(feature.name for feature in independentFeatures)
        dependent_feature_names = set(feature.name for feature in dependentFeatures)
        overlapping_features = independent_feature_names & dependent_feature_names
        if overlapping_features:
            msg = f"Input and output variables have the same name(s): {overlapping_features}"
            raise ValueError(msg)

        if meta is None:
            meta = dict()

        if organizations is None:
            organizations = []

        libraries = [
            Library(package_name, package_version)
            for package_name, package_version in get_installed_packages().items()
        ]

        jaqpotpyVersion = "1.0.0"
        # jaqpotpyVersion = str(jaqpotpy.__version__)

        data = {
            "meta": meta,
            "name": name,
            "description": description,
            "type": f"TORCH-{model_type}",
            "jaqpotpyVersion": jaqpotpyVersion,
            "libraries": [library.to_json() for library in libraries],
            "dependentFeatures": dependentFeatures,
            "independentFeatures": independentFeatures,
            "organizations": organizations,
            "visibility": visibility,
            "reliability": reliability,
            "pretrained": pretrained,
            "actualModel": actualModel,
            "additional_model_params": additional_model_params,
        }

        return data

    def set_input_feature_meta_for_deployment(self, feature_name: str, meta: dict):
        """
        Set the meta attribute for an input feature in the JSON data for deployment.

        Args:
            feature_name (str): The name of the input feature.
            meta (dict): The meta data to set for the input feature.
        """
        if not isinstance(feature_name, str):
            msg = "'feature_name' should be of type str"
            raise ValueError(msg)
        if not isinstance(meta, dict):
            msg = "'meta' should be of type dict"
            raise ValueError(msg)
        self._set_feature_attr_for_deployment(
            feature_name, "meta", dict(meta), is_input_feature=True
        )

    def set_output_feature_meta_for_deployment(self, feature_name: str, meta: dict):
        """
        Set the meta attribute for an output feature in the JSON data for deployment.

        Args:
            feature_name (str): The name of the output feature.
            meta (dict): The meta data to set for the output feature.
        """
        if not isinstance(feature_name, str):
            msg = "'feature_name' should be of type str"
            raise ValueError(msg)
        if not isinstance(meta, dict):
            msg = "'meta' should be of type dict"
            raise ValueError(msg)
        self._set_feature_attr_for_deployment(
            feature_name, "meta", dict(meta), is_input_feature=False
        )

    def set_input_feature_description_for_deployment(
        self, feature_name: str, description: str
    ):
        """
        Set the description attribute for an input feature in the JSON data for deployment.

        Args:
            feature_name (str): The name of the input feature.
            description (str): The description to set for the input feature.
        """
        if not isinstance(feature_name, str):
            msg = "'feature_name' should be of type str"
            raise ValueError(msg)
        if not isinstance(description, str):
            msg = "'description' should be of type str"
            raise ValueError(msg)
        self._set_feature_attr_for_deployment(
            feature_name, "description", str(description), is_input_feature=True
        )

    def set_output_feature_description_for_deployment(
        self, feature_name: str, description: str
    ):
        """
        Set the description attribute for an output feature in the JSON data for deployment.

        Args:
            feature_name (str): The name of the output feature.
            description (str): The description to set for the output feature.
        """
        if not isinstance(feature_name, str):
            msg = "'feature_name' should be of type str"
            raise ValueError(msg)
        if not isinstance(description, str):
            msg = "'description' should be of type str"
            raise ValueError(msg)
        self._set_feature_attr_for_deployment(
            feature_name, "description", str(description), is_input_feature=False
        )

    def _set_feature_attr_for_deployment(
        self,
        feature_name,
        attr_name,
        attr_value,
        is_input_feature=True,
    ):
        """
        Set an attribute for a feature in the JSON data for deployment.

        Args:
            feature_name (str): The name of the feature.
            attr_name (str): The name of the attribute.
            attr_value: The value of the attribute.
            is_input_feature (bool, optional): Whether the feature is an input feature. Default is True.
        """

        if self.json_data_for_deployment is None:
            msg = "No JSON data for deployment to set feature attributes. You may need to call 'prepare_data_for_deployment' first."
            raise RuntimeError(msg)

        feature_dependency = (
            "independentFeatures" if is_input_feature else "dependentFeatures"
        )

        for i, feature in enumerate(self.json_data_for_deployment[feature_dependency]):
            if feature_name == feature["name"]:
                self.json_data_for_deployment[feature_dependency][i][
                    attr_name
                ] = attr_value
                return

        raise ValueError(f"No feature named '{feature_name}'")

    def deploy_model(self, token):
        """
        Deploy the model to Jaqpot.

        Args:
            token (str): The authorization token.

        Returns:
            Response: The response from the Jaqpot server.
        """

        if self.json_data_for_deployment is None:
            no_json_err_msg = (
                "There are no JSON data to deploy. Note that prepare_for_deployment() must be called first to prepare the JSON to be deployed. "
                f"Please check the function documentation for the specific arguments needed for the current class '{self.__class__.__name__}'."
            )
            raise RuntimeError(no_json_err_msg)

        # TODO: Replace with the appropriate headers
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # TODO: Replace with the appropriate url for model uploading
        # url = "http://localhost:8006/api/v1/models/upload/" # for jaqpotpy-torch-inference repo
        url = "http://localhost:8002/upload/"  # for jaqpotpy-inference repo

        # TODO: Add whatever else is needed in the POST request
        response = requests.post(
            url, json=self.json_data_for_deployment, headers=headers
        )

        return response
