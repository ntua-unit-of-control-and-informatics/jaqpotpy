Module jaqpotpy_torch.trainers.multiclass_trainers.multiclass_fc_model_trainer
==============================================================================
Author: Ioannis Pitoskas
Contact: jpitoskas@gmail.com

Classes
-------

`MulticlassFCModelTrainer(model, n_epochs, optimizer, loss_fn, scheduler=None, device='cpu', use_tqdm=True, log_enabled=True, log_filepath=None)`
:   Trainer class for Multiclass Classification using a Fully Connected Network on tabular data.
    
    The MulticlassFCModelTrainer constructor.
    
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
    >>> from jaqpotpy.jaqpotpy_torch.models import FullyConnectedNetwork
    >>> from jaqpotpy.jaqpotpy_torch.trainers import MulticlassFCModelTrainer
    >>> 
    >>> num_classes = 10
    >>> model = FullyConnectedNetwork(input_dim=10,
    ...                               hidden_dims=[32, 32]
    ...                               output_dim=num_classes)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    >>> loss_fn = torch.nn.CrossEntropyLoss()
    >>>
    >>> trainer = MulticlassFCModelTrainer(model, n_epochs=50, optimizer=optimizer, loss_fn=loss_fn)
    ```

    ### Ancestors (in MRO)

    * jaqpotpy_torch.trainers.multiclass_trainers.multiclass_model_trainer.MulticlassModelTrainer
    * jaqpotpy_torch.trainers.base.torch_model_trainer.TorchModelTrainer
    * abc.ABC

    ### Class variables

    `MODEL_TYPE`
    :   'multiclass-fc-model'

    ### Methods

    `get_model_kwargs(self, data)`
    :   Fetch the model's keyword arguments.
        
        Args:
            data (tuple): Tuple returned by the Dataloader.
        
        Returns:
            dict: The required model kwargs. Set of keywords: {'x'}.

    `prepare_for_deployment(self, preprocessor, endpoint_name: str, name: str, description: Optional[str] = None, visibility: str = 'PUBLIC', reliability: Optional[int] = None, pretrained: bool = False, meta: dict = {})`
    :   Prepare the model for deployment on Jaqpot.
        
        Args:
            preprocessor (object): The preprocessor used to transform the tabular data before training the model.
            endpoint_name (str): The name of the endpoint for the deployed model.
            name (str): The name to be assigned to the deployed model.
            description (str, optional): A description for the model to be deployed. Default is None.
            visibility (str, optional): Visibility of the deployed model. Can be 'PUBLIC', 'PRIVATE' or 'ORG_SHARED'. Default is 'PUBLIC'.
            reliability (int, optional): The models reliability. Default is None.
            pretrained (bool, optional): Indicates if the model is pretrained. Default is False.
            meta (dict, optional): Additional metadata for the model. Default is an empty dictionary.
        
        Returns:
            dict: The data to be sent to the API of Jaqpot in JSON format.
                  Note that in this case, the '*additional_model_params*' key contains a nested dictionary with they keys: {'*preprocessor*'}.