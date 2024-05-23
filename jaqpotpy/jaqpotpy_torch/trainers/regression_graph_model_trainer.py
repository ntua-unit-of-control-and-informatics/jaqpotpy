from tqdm import tqdm
from .torch_model_trainer import TorchModelTrainer
import torch.nn.functional as F
import sklearn.metrics as metrics
import torch


class RegressionGraphModelTrainer(TorchModelTrainer):
    def __init__(
            self, 
            model, 
            n_epochs, 
            optimizer, 
            loss_fn, 
            device='cpu', 
            use_tqdm=True,
            log_enabled=True,
            log_file=None,
            normalization_mean=0.0,
            normalization_std=1.0
            ):
        
        super().__init__(
            model,
            n_epochs,
            optimizer,
            loss_fn,
            device,
            use_tqdm,
            log_enabled,
            log_file
            )
        
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
         
    def train_one_epoch(self, train_loader):

        running_loss = 0
        total_samples = 0

        tqdm_loader = tqdm(train_loader, desc=f'Epoch {self.current_epoch}/{self.n_epochs}') if self.use_tqdm else train_loader

        self.model.train()
        for _, data in enumerate(tqdm_loader):

            data = data.to(self.device)
            y_norm = self._normalize(data.y)

            self.optimizer.zero_grad()

            # outputs = self.model(x=data.x, edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr).squeeze(-1)
            outputs = self.model(x=data.x, edge_index=data.edge_index, batch=data.batch).squeeze(-1)
            loss = self.loss_fn(outputs.float(), y_norm.float())

            running_loss += loss.item() * data.y.size(0)
            total_samples += data.y.size(0)
        
            loss.backward()
            self.optimizer.step()

            if self.use_tqdm:
                tqdm_loader.set_postfix(loss=running_loss/total_samples)

        avg_loss = running_loss / len(train_loader.dataset)

        if self.use_tqdm:
            tqdm_loader.set_postfix(loss=running_loss)
            tqdm_loader.close()
            
        return avg_loss
    
    def evaluate(self, val_loader):
        """
        Evaluate the model's performance on the validation set.

        Args:
            val_loader (torch_geometric.loader.DataLoader): DataLoader for the validation dataset.
        Returns:
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
            
                data = data.to(self.device)
                y_norm = self._normalize(data.y)
                
                # outputs = model(x=data.x, edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr).squeeze(-1)
                outputs = self.model(x=data.x, edge_index=data.edge_index, batch=data.batch).squeeze(-1)
                
                all_preds.extend(outputs.tolist())
                all_true.extend(y_norm.tolist())
                
                loss = self.loss_fn(outputs.float(), y_norm.float())
                
                running_loss += loss.item() * data.y.size(0)
                total_samples += data.y.size(0)
            
            avg_loss = running_loss / len(val_loader.dataset)
        
        
        all_true = self._denormalize(torch.tensor(all_true)).tolist()
        all_preds = self._denormalize(torch.tensor(all_preds)).tolist()

        metrics_dict = self._compute_metrics(all_true, all_preds)
        metrics_dict['loss'] = avg_loss
            
        return avg_loss, metrics_dict

    def _compute_metrics(self, y_true, y_pred):
    
        explained_variance = metrics.explained_variance_score(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
        mse = metrics.mean_squared_error(y_true, y_pred)
        rmse = metrics.root_mean_squared_error(y_true, y_pred)
        mae = metrics.mean_absolute_error(y_true, y_pred)

        metrics_dict = {
            'explained_variance': explained_variance,
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }

        return metrics_dict
    
    def train(self, train_loader, val_loader=None):        
        for i in range(self.n_epochs):            
            self.current_epoch += 1
            train_loss = self.train_one_epoch(train_loader)
            _, train_metrics_dict = self.evaluate(train_loader)
            if self.log_enabled:
                self.log_metrics(train_loss, metrics_dict=train_metrics_dict, mode='train')
            if val_loader:
                val_loss, val_metrics_dict = self.evaluate(val_loader)
                if self.log_enabled:
                    self.log_metrics(val_loss, metrics_dict=val_metrics_dict, mode='val')

    def _normalize(self, x):
        return x.sub_(self.normalization_mean).div_(self.normalization_std)

    def _denormalize(self, x):
        return x.mul_(self.normalization_std).add_(self.normalization_mean)

    def predict(self, val_loader):
        all_preds = []
        self.model.eval()
        with torch.no_grad():
            for _, data in enumerate(val_loader):
            
                data = data.to(self.device)
                
                # outputs = model(x=data.x, edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr).squeeze(-1)
                outputs = self.model(x=data.x, edge_index=data.edge_index, batch=data.batch).squeeze(-1)
                
                all_preds.extend(outputs.tolist())

        all_preds = self._denormalize(torch.tensor(all_preds)).tolist()

        return all_preds
    
    def log_metrics(self, loss, metrics_dict, mode='train'):
        if mode=='train':
            epoch_logs = ' Train: '
        elif mode=='val':
            epoch_logs = ' Val:   '
        else:
            raise ValueError(f"Invalid mode '{mode}'")

        epoch_logs += f"loss={loss:.4f}"
        for metric, value in metrics_dict.items():

            if metric=='loss':
                continue

            epoch_logs += ' | '
            epoch_logs += f"{metric}={value:.4f}"

        self.logger.info(epoch_logs)
