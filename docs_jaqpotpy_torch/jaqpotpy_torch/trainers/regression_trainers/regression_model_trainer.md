Module jaqpotpy_torch.trainers.regression_trainers.regression_model_trainer
===========================================================================
Author: Ioannis Pitoskas (jpitoskas@gmail.com)

Classes
-------

`RegressionModelTrainer(model, n_epochs, optimizer, loss_fn, scheduler=None, device='cpu', use_tqdm=True, log_enabled=True, log_filepath=None, normalization_mean=0.0, normalization_std=1.0)`
:   Abstract trainer class for Regression models using PyTorch.
    
    Attributes:
        normalization_mean (float): Mean used to normalize the true values of the regression variables before model training. 
        normalization_std (float): Standard deviation used to normalize the true values of the regression variables before model training.  
    
    The RegressionModelTrainer constructor.
    
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
        normalization_mean (float, optional): Mean used to normalize the true values of the regression variables before model training. Default is 0.
        normalization_std (float, optinal): Standard deviation used to normalize the true values of the regression variables before model training. Default is 1.

    ### Ancestors (in MRO)

    * jaqpotpy_torch.trainers.base.torch_model_trainer.TorchModelTrainer
    * abc.ABC

    ### Descendants

    * jaqpotpy_torch.trainers.regression_trainers.regression_fc_model_trainer.RegressionFCModelTrainer
    * jaqpotpy_torch.trainers.regression_trainers.regression_graph_model_trainer.RegressionGraphModelTrainer
    * jaqpotpy_torch.trainers.regression_trainers.regression_graph_model_with_external_trainer.RegressionGraphModelWithExternalTrainer

    ### Methods

    `evaluate(self, val_loader)`
    :   Evaluate the model's performance on the validation set.
        
        Args:
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the validation dataset.
        Returns:
            float: Average loss over the validation dataset.
            dict: Dictionary containing evaluation metrics. The keys represent the metric names and the values are floats.

    `get_model_kwargs(self, data)`
    :   This abstract method should be implemented by subclasses to provide model-specific keyword arguments based on the data.
        
        Args:
            data: Whatever data the respective dataloader fetches.
        Returns:
            dict: The kwargs that the forward method of the respective model expects as input.

    `predict(self, val_loader)`
    :   Provide predictions on the validation set.
        
        Args:
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the validation dataset.
        Returns:
            list: List of predictions.