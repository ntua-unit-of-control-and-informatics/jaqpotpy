"""Author: Ioannis Pitoskas (jpitoskas@gmail.com)"""

from . import BinaryModelTrainer

import torch
import base64
from jaqpotpy.schemas import Feature
from typing import Optional
import inspect


class BinaryGraphModelTrainer(BinaryModelTrainer):
    """Trainer class for Binary Classification using Graph Neural Networks for SMILES and external features.

    Attributes
    ----------
        decision_threshold (float): Decision threshold for binary classification.

    """

    MODEL_TYPE = "binary-graph-model"
    """'binary-graph-model'"""

    @classmethod
    def get_model_type(cls):
        return cls.MODEL_TYPE

    def __init__(
        self,
        model,
        n_epochs,
        optimizer,
        loss_fn,
        scheduler=None,
        device="cpu",
        use_tqdm=True,
        log_enabled=True,
        log_filepath=None,
        decision_threshold=0.5,
    ):
        """The BinaryGraphModelTrainer constructor.

        Args:
        ----
            model (torch.nn.Module): The torch model to be trained.
            n_epochs (int): Number of training epochs.
            optimizer (torch.optim.Optimizer): The optimizer used for training the model.
            loss_fn (torch.nn.Module): The loss function used for training.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The scheduler used for adjusting the learning rate during training. Default is None.
            device (str, optional): The device on which to train the model. Default is 'cpu'.
            use_tqdm (bool, optional): Whether to use tqdm for progress bars. Default is True.
            log_enabled (bool, optional): Whether logging is enabled. Default is True.
            log_filepath (str or None, optional): Path to the log file. If None, logging is not saved to a file. Default is None.
            decision_threshold (float, optional): Decision threshold for binary classification. Default is 0.5.

        Example:
        -------
        ```
        >>> import torch
        >>> from jaqpotpy.jaqpotpy_torch.models import GraphAttentionNetwork
        >>> from jaqpotpy.jaqpotpy_torch.trainers import BinaryGraphModelTrainer
        >>>
        >>> model = GraphAttentionNetwork(input_dim=10,
        ...                               hidden_dims=[32, 32]
        ...                               edge_dim=5,
        ...                               output_dim=1)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        >>> loss_fn = torch.nn.BCEWithLogitsLoss()
        >>>
        >>> trainer = BinaryGraphModelTrainer(model, n_epochs=50, optimizer=optimizer, loss_fn=loss_fn)
        ```

        """
        super().__init__(
            model=model,
            n_epochs=n_epochs,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            device=device,
            use_tqdm=use_tqdm,
            log_enabled=log_enabled,
            log_filepath=log_filepath,
            decision_threshold=decision_threshold,
        )

    def get_model_kwargs(self, data):
        """Fetch the model's keyword arguments.

        Args:
        ----
            data (torch_geometric.data.Data): Data object returned as returned by the Dataloader

        Returns:
        -------
            dict: The required model kwargs. Set of keywords: {'*x*', '*edge_index*', '*batch*', '*edge_attr*'}. Note that '*edge_attr*' is only present if the model supports edge features.

        """
        kwargs = {}

        kwargs["x"] = data.x
        kwargs["edge_index"] = data.edge_index
        kwargs["batch"] = data.batch

        if "edge_attr" in inspect.signature(self.model.forward).parameters:
            kwargs["edge_attr"] = data.edge_attr

        return kwargs

    def prepare_for_deployment(
        self,
        featurizer,
        endpoint_name: str,
        name: str,
        description: Optional[str] = None,
        visibility: str = "PUBLIC",
        reliability: Optional[int] = None,
        pretrained: bool = False,
        meta: dict = dict(),
    ):
        """Prepare the model for deployment on Jaqpot.

        Args:
        ----
            featurizer (object): The featurizer used to transform the SMILES to graph representations before training the model.
            endpoint_name (str): The name of the endpoint for the deployed model.
            name (str): The name to be assigned to the deployed model.
            description (str, optional): A description for the model to be deployed. Default is None.
            visibility (str, optional): Visibility of the deployed model. Can be 'PUBLIC', 'PRIVATE' or 'ORG_SHARED'. Default is 'PUBLIC'.
            reliability (int, optional): The models reliability. Default is None.
            pretrained (bool, optional): Indicates if the model is pretrained. Default is False.
            meta (dict, optional): Additional metadata for the model. Default is an empty dictionary.

        Returns:
        -------
            dict: The data to be sent to the API of Jaqpot in JSON format.
                  Note that in this case, the '*additional_model_params*' key contains a nested dictionary with they keys: {'*decision_threshold*', '*featurizer*'}.

        """
        # self.model = self.model.cpu()
        # # Compile model and return a ScriptModule object (C++ Wrapper)
        # model_scripted = torch.jit.script(self.model)
        # model_buffer = io.BytesIO()
        # # First argument is scripted module, second argument is the binary buffer
        # torch.jit.save(model_scripted, model_buffer)
        # # Start reading at the beginning of the binary buffer
        # model_buffer.seek(0)
        # # Retrieve binary data from file, encode to base64 and convert base64 to UTF-8
        # model_scripted_base64 = base64.b64encode(model_buffer.getvalue()).decode('utf-8')

        if self.model.training:
            self.model.eval()
            self.model = self.model.cpu()

        dummy_smile = "CCC"
        dummy_input = featurizer.featurize(dummy_smile)
        x = dummy_input.x
        edge_index = dummy_input.edge_index
        batch = torch.zeros(x.shape[0], dtype=torch.int64)
        torch.onnx.export(
            self.model,  # model being run
            args=(x, edge_index, batch),
            f="model.onnx",
            input_names=["x", "edge_index", "batch"],
            dynamic_axes={"x": {0: "nodes"}, "edge_index": {1: "edges"}, "batch": [0]},
        )
        with open("model.onnx", "rb") as f:
            onnx_model_bytes = f.read()
        model_scripted_base64 = base64.b64encode(onnx_model_bytes).decode("utf-8")
        import os

        os.remove("model.onnx")

        # featurizer_buffer = io.BytesIO()
        # pickle.dump(featurizer, featurizer_buffer)
        # featurizer_buffer.seek(0)
        # featurizer_pickle_base64 = base64.b64encode(featurizer_buffer.getvalue()).decode('utf-8')

        featurizer_json = featurizer.get_json_rep()
        additional_model_params = {
            "decision_threshold": self.decision_threshold,
            "featurizer": featurizer_json,
        }

        independentFeatures = [
            Feature(
                name="SMILES",
                featureDependency="INDEPENDENT",
                possibleValues=[],
                featureType="SMILES",
            )
        ]

        dependentFeatures = [
            Feature(
                name=endpoint_name,
                featureDependency="DEPENDENT",
                possibleValues=["0", "1"],
                featureType="CATEGORICAL",
            )
        ]

        self.json_data_for_deployment = self._model_data_as_json(
            actualModel=model_scripted_base64,
            name=name,
            description=description,
            model_type=self.get_model_type(),
            visibility=visibility,
            independentFeatures=independentFeatures,
            dependentFeatures=dependentFeatures,
            additional_model_params=additional_model_params,
            reliability=reliability,
            pretrained=pretrained,
            meta=meta,
        )

        return self.json_data_for_deployment
