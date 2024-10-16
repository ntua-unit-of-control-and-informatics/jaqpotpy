from torch.utils.data import Dataset
import torch


class SmilesSeqDataset(Dataset):
    def __init__(self, molecules, y, vectorizer):
        # Dims must be (samples, length, features)
        self.vectorizer = vectorizer  # Vectorizer must be fitted
        self.X = vectorizer.transform(molecules)  # .permute(0, 2, 1)
        self.y = torch.tensor(y)

    def get_feature_dim(self):
        return self.X.shape[-1]

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
