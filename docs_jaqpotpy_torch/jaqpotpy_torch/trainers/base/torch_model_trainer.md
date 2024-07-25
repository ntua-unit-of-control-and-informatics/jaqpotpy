Module jaqpotpy_torch.trainers.base.torch_model_trainer
=======================================================
Author: Ioannis Pitoskas (jpitoskas@gmail.com)

Classes
-------

`TorchModelTrainer(model: torch.nn.modules.module.Module, n_epochs: int, optimizer: torch.optim.optimizer.Optimizer, loss_fn: torch.nn.modules.module.Module, scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None, device: str = 'cpu', use_tqdm: bool = True, log_enabled: bool = True, log_filepath: Optional[str] = None)`
:   An abstract class for training a model and deploying it on Jaqpot.
    
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

    * abc.ABC

    ### Descendants

    * jaqpotpy_torch.trainers.binary_trainers.binary_model_trainer.BinaryModelTrainer
    * jaqpotpy_torch.trainers.multiclass_trainers.multiclass_model_trainer.MulticlassModelTrainer
    * jaqpotpy_torch.trainers.regression_trainers.regression_model_trainer.RegressionModelTrainer

    ### Static methods

    `collect_subclass_info()`
    :   Collect information about all subclasses of the current class.
        
        Returns:
            dict: A dictionary containing information about the subclasses (model type, parent class, if is abstract class or not).

    `get_model_type()`
    :   Return the type of the model as a string.
        
        Returns:
            str: The model type.

    `get_subclass_model_types()`
    :   Return the list of all the types of models that inherit from TorchModelTrainer and are implemented in the jaqpotpy_torch.trainers submodule.
        
        Returns:
            list: A list of model types as strings.

    ### Methods

    `deploy_model(self, token)`
    :   Deploy the model to Jaqpot.
        
        Args:
            token (str): The authorization token.
        
        Returns:
            Response: The response from the Jaqpot server.

    `evaluate(self, val_loader)`
    :   Evaluate the model.
        
        Args:
            loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the evaluation dataset.

    `prepare_for_deployment(self, *args, **kwargs)`
    :   Prepare the model data in JSON format for deployment on Jaqpot.

    `set_input_feature_description_for_deployment(self, feature_name: str, description: str)`
    :   Set the description attribute for an input feature in the JSON data for deployment.
        
        Args:
            feature_name (str): The name of the input feature.
            description (str): The description to set for the input feature.

    `set_input_feature_meta_for_deployment(self, feature_name: str, meta: dict)`
    :   Set the meta attribute for an input feature in the JSON data for deployment.
        
        Args:
            feature_name (str): The name of the input feature.
            meta (dict): The meta data to set for the input feature.

    `set_output_feature_description_for_deployment(self, feature_name: str, description: str)`
    :   Set the description attribute for an output feature in the JSON data for deployment.
        
        Args:
            feature_name (str): The name of the output feature.
            description (str): The description to set for the output feature.

    `set_output_feature_meta_for_deployment(self, feature_name: str, meta: dict)`
    :   Set the meta attribute for an output feature in the JSON data for deployment.
        
        Args:
            feature_name (str): The name of the output feature.
            meta (dict): The meta data to set for the output feature.

    `train(self, train_loader, val_loader=None)`
    :   Train the model.
        
        Args:
            train_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the training dataset.            
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader], optional): DataLoader for the validation dataset.
        Returns:
            None

`TorchModelTrainerMeta(*args, **kwargs)`
:   Metaclass for defining Abstract Base Classes (ABCs).
    
    Use this metaclass to create an ABC.  An ABC can be subclassed
    directly, and then acts as a mix-in class.  You can also register
    unrelated concrete classes (even built-in classes) and unrelated
    ABCs as 'virtual subclasses' -- these and their descendants will
    be considered subclasses of the registering ABC by the built-in
    issubclass() function, but the registering ABC won't show up in
    their MRO (Method Resolution Order) nor will method
    implementations defined by the registering ABC be callable (not
    even via super()).

    ### Ancestors (in MRO)

    * abc.ABCMeta
    * builtins.type