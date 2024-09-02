"""Author: Ioannis Pitoskas (jpitoskas@gmail.com)"""

from ..base import TorchModelTrainer
from abc import abstractmethod
from tqdm import tqdm
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics


class BinaryModelTrainer(TorchModelTrainer):
    """Abstract trainer class for Binary Classification models using PyTorch.

    Attributes
    ----------
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
        decision_threshold=0.5,
    ):
        """The BinaryModelTrainer constructor.

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

        self.decision_threshold = decision_threshold

    @abstractmethod
    def get_model_kwargs(self, data):
        """This abstract method should be implemented by subclasses to provide model-specific keyword arguments based on the data.

        Args:
        ----
            data: Whatever data the respective dataloader fetches.

        Returns:
        -------
            dict: The kwargs that the forward method of the respective model expects as input.

        """
        pass

    def train(self, train_loader, val_loader=None):
        """Train the model.

        Args:
        ----
            train_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the training dataset.
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader], optional): DataLoader for the validation dataset.

        Returns:
        -------
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
        """This helper method handles the training loop for a single epoch, updating the model parameters and computing the running loss.

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
                preds = (probs > self.decision_threshold).int()

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

        #     tn, fp, fn, tp = metrics.confusion_matrix(all_labels, all_preds).ravel()
        #     conf_mat = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

        return avg_loss, metrics_dict, conf_mat

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

                probs = F.sigmoid(outputs)
                preds = (probs > self.decision_threshold).int()

                all_preds.extend(preds.tolist())

        return all_preds

    def predict_proba(self, val_loader):
        """Provide the probabilities of the predictions on the validation set.

        Args:
        ----
            val_loader (Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]): DataLoader for the validation dataset.

        Returns:
        -------
            list: List of predictions' probabilities.

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
