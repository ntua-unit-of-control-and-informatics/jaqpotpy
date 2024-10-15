from torch.utils.data import Dataset
import torch


class SMILESMolDataset(Dataset):
    def __init__(self, molecules, y, vectorizer):
        self.molecules = molecules
        self.y = y
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mols = self.molecules[idx]

        sample = self.vectorizer.transform([mols])[0]
        label = self.y[idx]
        return sample, label
