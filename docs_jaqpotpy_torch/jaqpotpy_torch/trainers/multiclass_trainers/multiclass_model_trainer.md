Module jaqpotpy_torch.trainers.multiclass_trainers.multiclass_model_trainer
===========================================================================
Author: Ioannis Pitoskas (jpitoskas@gmail.com)

Classes
-------

`MulticlassModelTrainer(model, n_epochs, optimizer, loss_fn, scheduler=None, device='cpu', use_tqdm=True, log_enabled=True, log_filepath=None)`
:   Abstract trainer class for Multiclass Classification models using PyTorch.
    
    The MulticlassModelTrainer constructor.
    
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

    ### Ancestors (in MRO)

    * jaqpotpy_torch.trainers.base.torch_model_trainer.TorchModelTrainer
    * abc.ABC

    ### Descendants

    * jaqpotpy_torch.trainers.multiclass_trainers.multiclass_fc_model_trainer.MulticlassFCModelTrainer
    * jaqpotpy_torch.trainers.multiclass_trainers.multiclass_graph_model_trainer.MulticlassGraphModelTrainer
    * jaqpotpy_torch.trainers.multiclass_trainers.multiclass_graph_model_with_external_trainer.MulticlassGraphModelWithExternalTrainer

    ### Methods

    `evaluate(self, val_loader)`
    :   Evaluate the model's performance on the validation set.
        
        Args:
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the validation dataset.
        Returns:
            float: Average loss over the validation dataset.
            dict: Dictionary containing evaluation metrics. The keys represent the metric names and the values are floats.
            numpy.ndarray: Confusion matrix C as a numpy.ndarray of shape (num_classes, num_classes), with Cij being equal to 
                           the number of observations known to be in group i and predicted to be in group j.

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

    `predict_proba(self, val_loader)`
    :   Provide the probabilities of the predictions on the validation set.
        
        Args:
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the validation dataset.
        Returns:
            list: List of predictions' probabilities.