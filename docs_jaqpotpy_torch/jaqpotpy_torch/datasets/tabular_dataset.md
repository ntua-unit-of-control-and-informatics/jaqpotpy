Module jaqpotpy_torch.datasets.tabular_dataset
==============================================
Author: Ioannis Pitoskas (jpitoskas@gmail.com)

Classes
-------

`TabularDataset(X, y=None)`
:   A PyTorch Dataset class for handling tabular data.
    
    Attributes:
        X (torch.tensor): A 2D tensor containing the feature data.
        y (torch.tensor): A 1D tensor containing the target data.
    
    The TabularDataset constructor.
    
    Args:
        X (numpy.ndarray or pandas.DataFrame): Feature data matrix of shape (n_samples, n_features).
        y (numpy.ndarray or pandas.DataFrame, optional): Target data of shape (n_samples,).
    
    Example:
    ```
    >>> import numpy as np
    >>> X = np.random.rand(3, 2)
    >>> y = np.random.rand(3, 2)
    >>> dataset = TabularDataset(X, y=y)
    >>> dataset[0]
    (tensor([0.7778, 0.3400]), tensor(0.4730))
    ```

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic

    ### Methods

    `get_num_features(self)`
    :   Returns the number of features in the dataset.
        
        Returns:
            int: Number of features.