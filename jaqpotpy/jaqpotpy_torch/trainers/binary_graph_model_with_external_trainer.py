from tqdm import tqdm
import torch.nn.functional as F
import sklearn.metrics as metrics
import torch
import base64
import io
import pickle

import torch_geometric

from .binary_graph_model_trainer import BinaryGraphModelTrainer


class BinaryGraphModelWithExternalTrainer(BinaryGraphModelTrainer):
    @property
    def model_type(self):
        return 'binary-graph-model-with-external'
    
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
            decision_threshold=0.5
            ):
        
        super().__init__(model, 
            n_epochs, 
            optimizer, 
            loss_fn, 
            device, 
            use_tqdm,
            log_enabled,
            log_file,
            decision_threshold
            )
        
    def train_one_epoch(self, train_loader):

        running_loss = 0
        total_samples = 0

        tqdm_loader = tqdm(train_loader, desc=f'Epoch {self.current_epoch}/{self.n_epochs}') if self.use_tqdm else train_loader

        self.model.train()
        for _, data in enumerate(tqdm_loader):

            data = data.to(self.device)

            self.optimizer.zero_grad()

            # outputs = self.model(x=data.x, edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr).squeeze(-1)
            outputs = self.model(x=data.x, edge_index=data.edge_index, external=data.external, batch=data.batch).squeeze(-1)
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
                outputs = self.model(x=data.x, edge_index=data.edge_index, external=data.external, batch=data.batch).squeeze(-1)

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

    def predict(self, val_loader):
        all_preds = []
        self.model.eval()
        with torch.no_grad():
            for _, data in enumerate(val_loader):
            
                data = data.to(self.device)
                # outputs = model(x=data.x, edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr).squeeze(-1)
                outputs = self.model(x=data.x, edge_index=data.edge_index, external=data.external, batch=data.batch).squeeze(-1)

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
                outputs = self.model(x=data.x, edge_index=data.edge_index, external=data.external, batch=data.batch).squeeze(-1)

                probs = F.sigmoid(outputs)
                
                all_probs.extend(probs.tolist())
        
        return all_probs

    def prepare_for_deployment(self,
                               featurizer,
                               external_feature_names,
                               endpoint_name,
                               external_normalization_mean,
                               external_normalization_std,
                               public=False,
                               meta=dict()
                               ):
        
        if len(external_feature_names) != self.model.num_external_features:
            raise ValueError(f"Size {len(external_feature_names)} of 'external_feature_names' must match the number of {self.model.num_external_features} external features that the model expects as input.")
        if 'SMILES' in external_feature_names:
            raise ValueError("The 'SMILES' keyword is in a protected namespace and should not be used for external features.")
        
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
            'decision_threshold': self.decision_threshold,
            'external_normalization_mean': external_normalization_mean,
            'external_normalization_std': external_normalization_std
            }
        

        libraries = [
            {'name': torch.__name__, 'version': str(torch.__version__)},
            {'name': torch_geometric.__name__, 'version': str(torch_geometric.__version__)}
        ]


        independentFeatures = ['SMILES'] + external_feature_names

        self.json_data_for_deployment = self._model_data_as_json(actualModel=model_scripted_base64,
                                                                 model_type=self.model_type,
                                                                 featurizer=featurizer_pickle_base64,
                                                                 public=public,
                                                                 libraries=libraries,
                                                                 independentFeatures=independentFeatures,
                                                                 dependentFeatures=[endpoint_name],
                                                                 additional_model_params=additional_model_params,
                                                                 reliability=0,
                                                                 pretrained=False,
                                                                 meta=meta
                                                                 )
        
        return self.json_data_for_deployment
