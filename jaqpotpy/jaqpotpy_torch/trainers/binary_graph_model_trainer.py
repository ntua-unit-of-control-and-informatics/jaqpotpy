from tqdm import tqdm
from .torch_model_trainer import TorchModelTrainer
import torch.nn.functional as F
import sklearn.metrics as metrics
import torch
import base64
import io
import pickle

import torch_geometric


class BinaryGraphModelTrainer(TorchModelTrainer):
    
    @property
    def model_type(self):
        return 'binary-graph-model'

    def __init__(
            self, 
            model, 
            n_epochs, 
            optimizer, 
            loss_fn, 
            device='cpu', 
            use_tqdm=True,
            log_enabled=True,
            log_filepath=None,
            decision_threshold=0.5,
            ):
        
        super().__init__(
            model,
            n_epochs,
            optimizer,
            loss_fn,
            device,
            use_tqdm,
            log_enabled,
            log_filepath
            )
        
        self.decision_threshold = decision_threshold
        self.log_enabled = log_enabled
        

    def train_one_epoch(self, train_loader):

        running_loss = 0
        total_samples = 0

        tqdm_loader = tqdm(train_loader, desc=f'Epoch {self.current_epoch}/{self.n_epochs}') if self.use_tqdm else train_loader

        self.model.train()
        for _, data in enumerate(tqdm_loader):

            data = data.to(self.device)

            self.optimizer.zero_grad()

            # outputs = self.model(x=data.x, edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr).squeeze(-1)
            outputs = self.model(x=data.x, edge_index=data.edge_index, batch=data.batch).squeeze(-1)
            loss = self.loss_fn(outputs.float(), data.y.float())

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
            numpy.ndarray: Confusion matrix as a numpy array of shape (4,) representing true negative (TN), false positive (FP), 
                           false negative (FN), and true positive (TP) counts respectively. The elements are arranged as [TN, FP, FN, TP].
        """
    
        running_loss = 0
        total_samples = 0
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for _, data in enumerate(val_loader):
            
                data = data.to(self.device)
                # outputs = model(x=data.x, edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr).squeeze(-1)
                outputs = self.model(x=data.x, edge_index=data.edge_index, batch=data.batch).squeeze(-1)

                probs = F.sigmoid(outputs)
                preds = (probs > self.decision_threshold).int()
                
                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_labels.extend(data.y.tolist())
                
                loss = self.loss_fn(outputs.float(), data.y.float())
                
                running_loss += loss.item() * data.y.size(0)
                total_samples += data.y.size(0)
            
            avg_loss = running_loss / len(val_loader.dataset)
        
        metrics_dict = self._compute_metrics(all_labels, all_preds)
        metrics_dict['roc_auc'] = metrics.roc_auc_score(all_labels, all_probs)
        metrics_dict['loss'] = avg_loss
        conf_mat = metrics.confusion_matrix(all_labels, all_preds).ravel()

        #     tn, fp, fn, tp = metrics.confusion_matrix(all_labels, all_preds).ravel()
        #     conf_mat = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        
        return avg_loss, metrics_dict, conf_mat
    
    def _compute_metrics(self, y_true, y_pred):
    
        accuracy = metrics.accuracy_score(y_true, y_pred)
        balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred, zero_division=0)
        recall = metrics.recall_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)
        mcc = metrics.matthews_corrcoef(y_true, y_pred)

        metrics_dict = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc
        }
        
        return metrics_dict
    
    def train(self, train_loader, val_loader=None):
        for i in range(self.n_epochs):            
            self.current_epoch += 1
            train_loss = self.train_one_epoch(train_loader)
            _, train_metrics_dict, train_conf_mat = self.evaluate(train_loader)
            if self.log_enabled:
                self.log_metrics(train_loss, metrics_dict=train_metrics_dict, mode='train')
            if val_loader:
                val_loss, val_metrics_dict, _ = self.evaluate(val_loader)
                if self.log_enabled:
                    self.log_metrics(val_loss, metrics_dict=val_metrics_dict, mode='val')

    
    def predict(self, val_loader):
        all_preds = []
        self.model.eval()
        with torch.no_grad():
            for _, data in enumerate(val_loader):
            
                data = data.to(self.device)
                # outputs = model(x=data.x, edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr).squeeze(-1)
                outputs = self.model(x=data.x, edge_index=data.edge_index, batch=data.batch).squeeze(-1)

                probs = F.sigmoid(outputs)
                preds = (probs > self.decision_threshold).int()
                
                all_preds.extend(preds.tolist())
        
        return all_preds
    
    def predict_proba(self, val_loader):
        all_probs = []
        self.model.eval()
        with torch.no_grad():
            for _, data in enumerate(val_loader):
            
                data = data.to(self.device)
                # outputs = model(x=data.x, edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr).squeeze(-1)
                outputs = self.model(x=data.x, edge_index=data.edge_index, batch=data.batch).squeeze(-1)

                probs = F.sigmoid(outputs)
                
                all_probs.extend(probs.tolist())
        
        return all_probs
    
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
    
    def prepare_for_deployment(self,
                               featurizer,
                               endpoint_name,
                               public=False,
                               meta=dict()
                               ):
        
        model_scripted = torch.jit.script(self.model)
        model_buffer = io.BytesIO()
        torch.jit.save(model_scripted, model_buffer)
        model_buffer.seek(0)
        model_scripted_base64 = base64.b64encode(model_buffer.getvalue()).decode('utf-8')


        featurizer_buffer = io.BytesIO()
        pickle.dump(featurizer, featurizer_buffer)
        featurizer_buffer.seek(0)
        featurizer_pickle_base64 = base64.b64encode(featurizer_buffer.getvalue()).decode('utf-8')
        
        additional_model_params = {
            'decision_threshold': self.decision_threshold
            }
        

        libraries = [
            {'name': torch.__name__, 'version': str(torch.__version__)},
            {'name': torch_geometric.__name__, 'version': str(torch_geometric.__version__)}
        ]

        self.json_data_for_deployment = self._model_data_as_json(actualModel=model_scripted_base64,
                                                                 model_type=self.model_type,
                                                                 featurizer=featurizer_pickle_base64,
                                                                 public=public,
                                                                 libraries=libraries,
                                                                 independentFeatures=['SMILES'],
                                                                 dependentFeatures=[endpoint_name],
                                                                 additional_model_params=additional_model_params,
                                                                 reliability=0,
                                                                 pretrained=False,
                                                                 meta=meta
                                                                 )
        
        return self.json_data_for_deployment

    

    # def save(self):
    #     task = 'binary'

    #     with open(f'metadata_{task}.json', 'w') as f:
    #         metadata = {
    #             "task": task,
    #             "decision_threshold": self.decision_threshold,
    #             }
    #         import json
    #         json.dump(metadata, f)
            
    #     model_scripted = torch.jit.script(self.model)
    #     model_scripted.save(f'model_scripted_{task}.pt')
