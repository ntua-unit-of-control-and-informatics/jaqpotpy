Module jaqpotpy_torch.trainers.regression_trainers.regression_graph_model_trainer
=================================================================================
Author: Ioannis Pitoskas (jpitoskas@gmail.com)

Classes
-------

`RegressionGraphModelTrainer(model, n_epochs, optimizer, loss_fn, scheduler=None, device='cpu', use_tqdm=True, log_enabled=True, log_filepath=None, normalization_mean=0.5, normalization_std=1.0)`
:   Trainer class for Regression using Graph Neural Networks for SMILES and external features. 
    
    The RegressionGraphModelTrainer constructor.
    
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
        normalization_std' (float, optinal): Standard deviation used to normalize the true values of the regression variables before model training. Default is 1. 
    
    Example:
    ```
    >>> import torch
    >>> from jaqpotpy.jaqpotpy_torch.models import GraphAttentionNetwork
    >>> from jaqpotpy.jaqpotpy_torch.trainers import RegressionGraphModelTrainer
    >>> 
    >>> model = GraphAttentionNetwork(input_dim=10,
    ...                               hidden_dims=[32, 32]
    ...                               edge_dim=5,
    ...                               output_dim=num_classes)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    >>> loss_fn = torch.nn.MSELoss()
    >>>
    >>> trainer = MulticlassGraphModelTrainer(model, n_epochs=50, optimizer=optimizer, loss_fn=loss_fn)
    ```

    ### Ancestors (in MRO)

    * jaqpotpy_torch.trainers.regression_trainers.regression_model_trainer.RegressionModelTrainer
    * jaqpotpy_torch.trainers.base.torch_model_trainer.TorchModelTrainer
    * abc.ABC

    ### Class variables

    `MODEL_TYPE`
    :   'regression-graph-model'

    ### Methods

    `get_model_kwargs(self, data)`
    :   Fetch the model's keyword arguments.
        
        Args:
            data (torch_geometric.data.Data): Data object returned as returned by the Dataloader
        
        Returns:
            dict: The required model kwargs. Set of keywords: {'*x*', '*edge_index*', '*batch*', '*edge_attr*'}. Note that '*edge_attr*' is only present if the model supports edge features.

    `prepare_for_deployment(self, featurizer, endpoint_name: str, name: str, description: Optional[str] = None, visibility: str = 'PUBLIC', reliability: Optional[int] = None, pretrained: bool = False, meta: dict = {})`
    :   Args:
            featurizer (object): The featurizer used to transform the SMILES to graph representations before training the model.
            endpoint_name (str): The name of the endpoint for the deployed model.
            name (str): The name to be assigned to the deployed model.
            description (str, optional): A description for the model to be deployed. Default is None.
            visibility (str, optional): Visibility of the deployed model. Can be 'PUBLIC', 'PRIVATE' or 'ORG_SHARED'. Default is 'PUBLIC'.
            reliability (int, optional): The models reliability. Default is None.
            pretrained (bool, optional): Indicates if the model is pretrained. Default is False.
            meta (dict, optional): Additional metadata for the model. Default is an empty dictionary.
        
        Returns:
            dict: The data to be sent to the API of Jaqpot in JSON format.
                  Note that in this case, the '*additional_model_params*' key contains a nested dictionary with they keys: {'*normalization_mean*', '*normalization_std*', '*featurizer*'}.