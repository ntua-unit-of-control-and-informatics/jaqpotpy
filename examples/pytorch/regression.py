import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from jaqpot_api_client import ModelTask, Feature, FeatureType, ModelVisibility
from jaqpotpy.models.torch_models.torch_onnx import TorchONNXModel
from jaqpotpy import Jaqpot


# Create a simple dataset class for regression
class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Define the neural network model
class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)


def main():
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 5)  # 1000 samples, 5 features
    y = (
        2 * X[:, 0]
        + 3 * X[:, 1]
        - X[:, 2]
        + 0.5 * X[:, 3]
        + np.random.randn(1000) * 0.1
    )

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)

    # Create dataset and dataloader
    dataset = RegressionDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = RegressionNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    n_epochs = 100
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(train_loader):.4f}"
            )

    # Set model to evaluation mode
    model.eval()

    # Prepare for Jaqpot deployment
    # Create example input tensor
    input_tensor = torch.randn(1, 5)  # Batch size 1, 5 features

    # Define features
    independent_features = [
        Feature(key=f"feature_{i}", name=f"Feature {i}", feature_type=FeatureType.FLOAT)
        for i in range(5)
    ]
    dependent_features = [
        Feature(key="target", name="Target", feature_type=FeatureType.FLOAT)
    ]

    # Create ONNX model wrapper
    jaqpot_model = TorchONNXModel(
        model=model,
        input_example=input_tensor,
        task=ModelTask.REGRESSION,
        independent_features=independent_features,
        dependent_features=dependent_features,
        onnx_preprocessor=None,
    )

    # Initialize Jaqpot and deploy model
    jaqpot = Jaqpot()
    jaqpot.login()  # This will open a browser window for authentication

    jaqpot_model.deploy_on_jaqpot(
        jaqpot=jaqpot,
        name="PyTorch Regression Model",
        description="A simple neural network for regression with 5 input features",
        visibility=ModelVisibility.PUBLIC,
    )


if __name__ == "__main__":
    main()
