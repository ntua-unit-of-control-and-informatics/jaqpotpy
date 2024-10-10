import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer

from jaqpotpy import Jaqpot
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors.molecular import (
    MACCSKeysFingerprint,
)
from jaqpotpy.models import SklearnModel

# path = "../jaqpotpy/test_data/test_data_smiles_classification.csv"
path = "jaqpotpy/test_data/test_data_smiles_classification.csv"


df = pd.read_csv(path).iloc[0:100, :]
smiles_cols = ["SMILES"]
y_cols = ["ACTIVITY"]
x_cols = ["X1", "X2"]
featurizer = MACCSKeysFingerprint()
dataset = JaqpotpyDataset(
    df=df,
    y_cols=y_cols,
    smiles_cols=smiles_cols,
    x_cols=x_cols,
    task="regression",
    featurizer=featurizer,
)


model = RandomForestRegressor(random_state=42)

molecularModel_t1 = SklearnModel(
    dataset=dataset,
    model=model,
    preprocess_x=ColumnTransformer(
        transformers=[
            # ("Standard Scaler", StandardScaler(), ["X1", "X2"]),
            ("OneHotEncoder", OneHotEncoder(), ["Cat_col"]),
        ],
        remainder="passthrough",
    ),
)
