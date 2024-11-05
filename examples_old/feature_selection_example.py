import pandas as pd

from jaqpotpy.descriptors.molecular import TopologicalFingerprint, RDKitDescriptors
from jaqpotpy.datasets import JaqpotpyDataset
from sklearn.feature_selection import VarianceThreshold


path = "./jaqpotpy/test_data/test_data_smiles_regression.csv"

# Example 1: Select features using sklearn.feature_selection
df = pd.read_csv(path).iloc[0:5, :]
smiles_cols = ["SMILES"]
y_cols = ["ACTIVITY"]
x_cols = ["X1", "X2"]
featurizer = [TopologicalFingerprint(), RDKitDescriptors()]
dataset = JaqpotpyDataset(
    df=df,
    y_cols=y_cols,
    smiles_cols=smiles_cols,
    x_cols=x_cols,
    task="regression",
    featurizer=featurizer,
)
print(dataset.X.shape)
FeatureSelector = VarianceThreshold(threshold=0.8)
dataset.select_features(FeatureSelector)
print(dataset.X.shape)
print(dataset.active_features)

# Example 2: Select features using sklearn.feature_selection
df = pd.read_csv(path).iloc[0:5, :]
smiles_cols = ["SMILES"]
y_cols = ["ACTIVITY"]
x_cols = ["X1", "X2"]
featurizer = [TopologicalFingerprint(), RDKitDescriptors()]
dataset = JaqpotpyDataset(
    df=df,
    y_cols=y_cols,
    smiles_cols=smiles_cols,
    x_cols=x_cols,
    task="regression",
    featurizer=featurizer,
)
print(dataset.X.shape)
dataset.select_features(SelectionList=["X1"])
print(dataset.X.shape)
print(dataset.active_features)
