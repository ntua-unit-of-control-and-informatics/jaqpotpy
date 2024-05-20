from abc import ABC, abstractmethod
import torch
import logging
import sys

class TorchModelTrainer(ABC):
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
        self.log_filepath = log_filepath

        self.logger = self._setup_logger()

    def _setup_logger(self):

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
        handlers=[logging.StreamHandler(sys.stdout)]
        if self.log_filepath:
            log_file_handler = logging.FileHandler(self.log_filepath)
            handlers.append(log_file_handler)

        log_format = '%(message)s'
        formatter = logging.Formatter(log_format)
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

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