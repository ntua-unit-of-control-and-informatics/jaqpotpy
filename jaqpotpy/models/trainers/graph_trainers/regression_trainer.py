from jaqpotpy.api.openapi.models.feature import Feature
from jaqpotpy.api.openapi.models.feature_type import FeatureType
from tqdm import tqdm
import torch
import io
import base64
from typing import Optional
import inspect
from jaqpotpy.models.trainers import TorchModelTrainer
from sklearn import metrics
import torch.nn.functional as F


class RegressionGraphModelTrainer(TorchModelTrainer):
    """
    Trainer class for Regression using Graph Neural Networks for SMILES
    """

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
        # normalization_mean=0.5,
        # normalization_std=1.0,
    ):
        """The RegressionGraphModelTrainer constructor.

        Args:
        ----
            model (torch.nn.Module): The torch model to be trained.
            n_epochs (int): Number of training epochs.
            optimizer (torch.optim.Optimizer): The optimizer used for training the model.
            loss_fn (torch.nn.Module): The loss function used for training.
            scheduler (torch.optim.lr_scheduler.LRScheduler, optional): The scheduler used for adjusting the learning rate during training. Default is None.
            device (str, optional): The device on which to train the model. Default is 'cpu'.
            use_tqdm (bool, optional): Whether to use tqdm for progress bars. Default is True.
            log_enabled (bool, optional): Whether logging is enabled. Default is True.
            log_filepath (str or None, optional): Path to the log file. If None, logging is not saved to a file. Default is None.
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
        )

    def get_model_kwargs(self, data):
        """Fetch the model's keyword arguments.

        Args:
        ----
            data (torch_geometric.data.Data): Data object returned from the DataLoader.

        Returns:
        -------
            dict: The required model keyword arguments. Contains keys: {'x', 'edge_index', 'batch', 'edge_attr'}.
                  Note that 'edge_attr' is only present if the model supports edge features.
        """
        kwargs = {}

        kwargs["x"] = data.x
        kwargs["edge_index"] = data.edge_index
        kwargs["batch"] = data.batch

        if "edge_attr" in inspect.signature(self.model.forward).parameters:
            kwargs["edge_attr"] = data.edge_attr

        return kwargs

    def train(self, train_loader, val_loader=None):
        """Train the model.

        Args:
        ----
            train_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]):
                DataLoader for the training dataset.
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader], optional):
                DataLoader for the validation dataset. Default is None.

        Returns:
        -------
            None
        """
        for i in range(self.n_epochs):
            self.current_epoch += 1

            train_loss = self._train_one_epoch(train_loader)
            _, train_metrics_dict = self.evaluate(train_loader)

            if self.log_enabled:
                self.logger.info(f"Epoch {self.current_epoch}:")
                self._log_metrics(
                    train_loss, metrics_dict=train_metrics_dict, mode="train"
                )

            if val_loader:
                val_loss, val_metrics_dict = self.evaluate(val_loader)
                if self.log_enabled:
                    self._log_metrics(
                        val_loss, metrics_dict=val_metrics_dict, mode="val"
                    )

            self.scheduler.step()

    def _train_one_epoch(self, train_loader):
        """Handles the training loop for a single epoch, updating the model parameters and computing the running loss.

        Args:
        ----
            train_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the training dataset.

        Returns:
        -------
            float: The average loss of the current epoch.
        """
        running_loss = 0
        total_samples = 0

        tqdm_loader = (
            tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.n_epochs}")
            if self.use_tqdm
            else train_loader
        )

        self.model.train()
        for _, data in enumerate(tqdm_loader):
            try:  # data might come from torch_geomtric Dataloader or from torch.utils.Dataloader
                data = data.to(self.device)
                y = data.y
            except AttributeError:
                data = [d.to(self.device) for d in data]
                y = data[-1]

            model_kwargs = self.get_model_kwargs(data)

            self.optimizer.zero_grad()

            outputs = self.model(**model_kwargs).squeeze(-1)
            loss = self.loss_fn(outputs.float(), y.float())

            running_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

            loss.backward()
            self.optimizer.step()

            if self.use_tqdm:
                tqdm_loader.set_postfix(loss=running_loss / total_samples)

        avg_loss = running_loss / len(train_loader.dataset)

        if self.use_tqdm:
            tqdm_loader.set_postfix(loss=running_loss)
            tqdm_loader.close()

        return avg_loss

    def evaluate(self, val_loader):
        """Evaluate the model's performance on the validation set.

        Args:
        ----
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the validation dataset.

        Returns:
        -------
            float: Average loss over the validation dataset.
            dict: Dictionary containing evaluation metrics. The keys represent the metric names and the values are floats.
        """
        running_loss = 0
        total_samples = 0

        all_preds = []
        all_true = []

        self.model.eval()
        with torch.no_grad():
            for _, data in enumerate(val_loader):
                try:  # data might come from torch_geomtric Dataloader or from torch.utils.Dataloader
                    data = data.to(self.device)
                    y = data.y
                except AttributeError:
                    data = [d.to(self.device) for d in data]
                    y = data[-1]

                model_kwargs = self.get_model_kwargs(data)

                outputs = self.model(**model_kwargs).squeeze(-1)
                all_preds.extend(outputs.tolist())
                all_true.extend(y.tolist())

                loss = self.loss_fn(outputs.float(), y.float())

                running_loss += loss.item() * y.size(0)
                total_samples += y.size(0)

            avg_loss = running_loss / len(val_loader.dataset)

        all_true = torch.tensor(all_true).tolist()
        all_preds = torch.tensor(all_preds).tolist()

        metrics_dict = self._compute_metrics(all_true, all_preds)
        metrics_dict["loss"] = avg_loss

        return avg_loss, metrics_dict

    def predict(self, val_loader):
        """Provide predictions on the validation set.

        Args:
        ----
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the validation dataset.

        Returns:
        -------
            list: List of predictions.
        """
        all_preds = []
        self.model.eval()
        with torch.no_grad():
            for _, data in enumerate(val_loader):
                try:  # data might come from torch_geomtric Dataloader or from torch.utils.Dataloader
                    data = data.to(self.device)
                except AttributeError:
                    data = [d.to(self.device) for d in data]

                model_kwargs = self.get_model_kwargs(data)

                outputs = self.model(**model_kwargs).squeeze(-1)

                all_preds.extend(outputs.tolist())

        # all_preds = self._denormalize(torch.tensor(all_preds)).tolist()

        return all_preds

    def _log_metrics(self, loss, metrics_dict, mode="train"):
        """Log metrics for the current epoch.

        Args:
        ----
            loss (float): The loss value for the current epoch.
            metrics_dict (dict): Dictionary of metrics to log.
            mode (str, optional): Indicates whether the metrics are for training or validation. Default is "train".

        Returns:
        -------
            None
        """
        if mode == "train":
            epoch_logs = " Train: "
        elif mode == "val":
            epoch_logs = " Val:   "
        else:
            raise ValueError(f"Invalid mode '{mode}'")

        epoch_logs += f"loss={loss:.4f}"
        for metric, value in metrics_dict.items():
            if metric == "loss":
                continue

            epoch_logs += " | "
            epoch_logs += f"{metric}={value:.4f}"

        self.logger.info(epoch_logs)

    def _compute_metrics(self, y_true, y_pred):
        """Compute evaluation metrics based on the true and predicted values.

        Args:
        ----
            y_true (list): List of true values.
            y_pred (list): List of predicted values.

        Returns:
        -------
            dict: A dictionary containing the computed metrics such as explained variance, R^2, MSE, RMSE, and MAE.
        """
        explained_variance = metrics.explained_variance_score(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
        mse = metrics.mean_squared_error(y_true, y_pred)
        rmse = metrics.root_mean_squared_error(y_true, y_pred)
        mae = metrics.mean_absolute_error(y_true, y_pred)

        metrics_dict = {
            "explained_variance": explained_variance,
            "r2": r2,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
        }

        return metrics_dict

    # def _normalize(self, x):
    #     return x.sub_(self.normalization_mean).div_(self.normalization_std)

    # def _denormalize(self, x):
    #     return x.mul_(self.normalization_std).add_(self.normalization_mean)
