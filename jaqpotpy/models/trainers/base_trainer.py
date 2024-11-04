from abc import ABC, abstractmethod
import torch
import logging
import sys
import os
from typing import Optional
import inspect
from torch.optim.lr_scheduler import LambdaLR


class TorchModelTrainer(ABC):
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
        logger (logging.Logger): The logger object at INFO level used for logging during model training.
    """

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

        self.logger = self._setup_logger()

        self.model.to(self.device)

    def _setup_logger(self):
        """
        Sets up and returns a logger for the training process.

        Configures the logger to write to stdout and optionally to a file.
        Ensures existing handlers are removed to avoid duplicate logs.

        Returns:
            logging.Logger: Configured logger instance.

        Raises:
            ValueError: If the specified log file already exists.
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

    # @classmethod
    # def collect_subclass_info(cls):
    #     """
    #     Collects metadata about all subclasses of the TorchModelTrainer class.

    #     This method traverses subclasses to gather information, including model type, parent class,
    #     and whether each subclass is abstract.

    #     Returns:
    #         dict: Dictionary containing metadata for each subclass, with the following keys:
    #             - 'model_type': Model type of the subclass.
    #             - 'parent': Name of the parent class.
    #             - 'is_abstract': Boolean indicating if the subclass is abstract.
    #     """
    #     subclass_info = {}

    #     for subclass in cls.__subclasses__():
    #         subclass_info[subclass.__name__] = {
    #             "model_type": subclass.get_model_type(),
    #             "parent": cls.__name__,
    #             "is_abstract": inspect.isabstract(subclass),
    #         }

    #         subclass_info.update(subclass.collect_subclass_info())

    #     return subclass_info

    # @classmethod
    # def get_subclass_model_types(cls):
    #     """
    #     Retrieves a list of model types for all concrete (non-abstract) subclasses of TorchModelTrainer.

    #     Filters out abstract subclasses and collects the 'model_type' for each concrete subclass
    #     implemented in the `jaqpotpy_torch.trainers` submodule.

    #     Returns:
    #         list of str: A list of model types as strings.
    #     """
    #     return [
    #         v["model_type"]
    #         for v in cls.collect_subclass_info().values()
    #         if not v["is_abstract"] and v["model_type"] is not None
    #     ]

    # @check_reach_epoch_limit
    @abstractmethod
    def train(self, train_loader, val_loader=None):
        """
        Abstract method for training the model.

        Args:
            train_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]):
                DataLoader for the training dataset, providing batches of training data.
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader], optional):
                DataLoader for the validation dataset, providing batches of validation data.

        Returns:
            None

        Raises:
            NotImplementedError: If this method is not implemented in a subclass.
        """
        pass

    @abstractmethod
    def evaluate(self, val_loader):
        """
        Abstract method for evaluating the model.

        Args:
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]):
                DataLoader for the evaluation dataset, providing batches of data to evaluate model performance.

        Returns:
            None

        Raises:
            NotImplementedError: If this method is not implemented in a subclass.
        """
        pass
