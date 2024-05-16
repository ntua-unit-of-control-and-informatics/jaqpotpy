from abc import ABC, abstractmethod

class TorchModelTrainer(ABC):
    def __init__(self, model_architecture, optimizer, loss_fn):
        self.model_architecture = model_architecture
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
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
    def eval(self, val_loader):
        """
        Evaluate the model.

        Args:
            loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the evaluation dataset.
        Returns:
            None
        """
        pass