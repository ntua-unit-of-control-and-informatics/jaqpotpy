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


class BinarySequenceTrainer(TorchModelTrainer):
    """
    Trainer class for Binary Classification using LSTM Networks for SMILES.

    Attributes:
        decision_threshold (float): Decision threshold for binary classification.
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
        The BinaryGraphModelTrainer constructor.

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
            decision_threshold (float, optional): Decision threshold for binary classification. Default is 0.5.

        Example:
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
        )

    def train(self, train_loader, val_loader=None):
        """
        Train the model.

        Args:
            train_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the training dataset.
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader], optional): DataLoader for the validation dataset.
        Returns:
            None
        """

        for _ in range(self.n_epochs):
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
        This helper method handles the training loop for a single epoch, updating the model parameters and computing the running loss.

        Args:
            train_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the training dataset.
        Returns:
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
        for inputs, y in tqdm_loader:
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs).squeeze(-1)
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
        Evaluate the model's performance on the validation set.

        Args:
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the validation dataset.
        Returns:
            float: Average loss over the validation dataset.
            dict: Dictionary containing evaluation metrics. The keys represent the metric names and the values are floats.
            numpy.ndarray: Confusion matrix as a numpy array of shape (2, 2) representing true negative (TN), false positive (FP),
                           false negative (FN), and true positive (TP) counts respectively. The elements are arranged as [[TN, FP], [FN, TP]].
        """

        running_loss = 0
        total_samples = 0

        all_preds = []
        all_probs = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for inputs, y in val_loader:
                inputs = inputs.to(self.device)

                outputs = self.model(inputs).squeeze(-1)

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
        Provide predictions on the validation set.

        Args:
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the validation dataset.
        Returns:
            list: List of predictions.
        """
        all_preds = []
        self.model.eval()
        with torch.no_grad():
            for inputs, y in val_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs).squeeze(-1)

                probs = F.sigmoid(outputs)
                preds = (probs > 0.5).int()

                all_preds.extend(preds.tolist())

        return all_preds

    def predict_proba(self, val_loader):
        """
        Provide the probabilities of the predictions on the validation set.

        Args:
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the validation dataset.
        Returns:
            list: List of predictions' probabilities.
        """
        all_probs = []
        self.model.eval()
        with torch.no_grad():
            for inputs, y in val_loader:
                inputs = inputs.to(self.device)

                outputs = self.model(inputs).squeeze(-1)

                probs = F.sigmoid(outputs)

                all_probs.extend(probs.tolist())

        return all_probs

    def _log_metrics(self, loss, metrics_dict, mode="train"):
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
