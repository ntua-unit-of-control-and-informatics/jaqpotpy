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


class BinaryGraphModelTrainer(TorchModelTrainer):
    """
    Trainer class for binary classification using Graph Neural Networks (GNNs) designed for SMILES.
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
    ):
        """
        Initializes BinaryGraphModelTrainer for training binary classification GNNs.

        Args:
            model (torch.nn.Module): The torch model to be trained.
            n_epochs (int): Number of training epochs.
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            loss_fn (torch.nn.Module): Loss function for binary classification.
            scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Learning rate scheduler. Default is None.
            device (str, optional): Device for model training, e.g., 'cpu' or 'cuda'. Default is 'cpu'.
            use_tqdm (bool, optional): If True, uses tqdm for progress display. Default is True.
            log_enabled (bool, optional): If True, enables logging. Default is True.
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
        """
        Returns model's keyword arguments.

        Args:
            data (torch_geometric.data.Data): Data object returned from the DataLoader.

        Returns:
            dict: Model keyword arguments for graph data. Keys include 'x', 'edge_index', 'batch', and optionally 'edge_attr' if the model uses edge features.
        """

        kwargs = {}

        kwargs["x"] = data.x
        kwargs["edge_index"] = data.edge_index
        kwargs["batch"] = data.batch

        if "edge_attr" in inspect.signature(self.model.forward).parameters:
            kwargs["edge_attr"] = data.edge_attr

        return kwargs

    def train(self, train_loader, val_loader=None):
        """
        Trains the model for the specified number of epochs.

        Args:
            train_loader (torch_geometric.loader.DataLoader): DataLoader for training data.
            val_loader (torch_geometric.loader.DataLoader, optional): DataLoader for validation data.

        Returns:
            None
        """

        for i in range(self.n_epochs):
            self.current_epoch += 1

            train_loss = self._train_one_epoch(train_loader)
            _, train_metrics_dict, _ = self.evaluate(train_loader)

            if self.log_enabled:
                self._log_metrics(
                    train_loss, metrics_dict=train_metrics_dict, mode="train"
                )

            if val_loader:
                val_loss, val_metrics_dict, _ = self.evaluate(val_loader)
                if self.log_enabled:
                    self._log_metrics(
                        val_loss, metrics_dict=val_metrics_dict, mode="val"
                    )

            self.scheduler.step()

    def _train_one_epoch(self, train_loader):
        """
        Trains the model for a single epoch.

        Args:
            train_loader (torch_geometric.loader.DataLoader): DataLoader for training data.

        Returns:
            float: Average loss of the current epoch.
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
        """
        Evaluates the model on the validation set.

        Args:
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for validation data.

        Returns:
            float: Average loss on the validation dataset.
            dict: Dictionary of evaluation metrics (e.g., accuracy, precision).
            numpy.ndarray: Confusion matrix of shape (2, 2) representing TN, FP, FN, and TP counts.
        """

        running_loss = 0
        total_samples = 0

        all_preds = []
        all_probs = []
        all_labels = []

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

                probs = F.sigmoid(outputs)
                preds = (probs > 0.5).int()

                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_labels.extend(y.tolist())

                loss = self.loss_fn(outputs.float(), y.float())

                running_loss += loss.item() * y.size(0)
                total_samples += y.size(0)

            avg_loss = running_loss / len(val_loader.dataset)

        metrics_dict = self._compute_metrics(all_labels, all_preds)
        metrics_dict["roc_auc"] = metrics.roc_auc_score(all_labels, all_probs)
        metrics_dict["loss"] = avg_loss
        conf_mat = metrics.confusion_matrix(all_labels, all_preds)

        return avg_loss, metrics_dict, conf_mat

    def predict(self, val_loader):
        """
        Returns binary predictions for the validation set.

        Args:
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for validation data.

        Returns:
            list: List of binary predictions.
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

                probs = F.sigmoid(outputs)
                preds = (probs > 0.5).int()

                all_preds.extend(preds.tolist())

        return all_preds

    def predict_proba(self, val_loader):
        """
        Returns prediction probabilities for the validation set.

        Args:
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for validation data.

        Returns:
            list: List of prediction probabilities.
        """
        all_probs = []
        self.model.eval()
        with torch.no_grad():
            for _, data in enumerate(val_loader):
                try:  # data might come from torch_geomtric Dataloader or from torch.utils.Dataloader
                    data = data.to(self.device)
                except AttributeError:
                    data = [d.to(self.device) for d in data]

                model_kwargs = self.get_model_kwargs(data)

                outputs = self.model(**model_kwargs).squeeze(-1)

                probs = F.sigmoid(outputs)

                all_probs.extend(probs.tolist())

        return all_probs

    def _log_metrics(self, loss, metrics_dict, mode="train"):
        """
        Logs model metrics.

        Args:
            loss (float): Loss value.
            metrics_dict (dict): Dictionary of metric names and values.
            mode (str): "train" or "val" to indicate the logging mode.

        Returns:
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
        """
        Computes evaluation metrics.

        Args:
            y_true (list): True labels.
            y_pred (list): Predicted labels.

        Returns:
            dict: Dictionary of computed metrics.
        """
        accuracy = metrics.accuracy_score(y_true, y_pred)
        balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred, zero_division=0)
        recall = metrics.recall_score(y_true, y_pred, zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
        mcc = metrics.matthews_corrcoef(y_true, y_pred)

        metrics_dict = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc,
        }

        return metrics_dict
