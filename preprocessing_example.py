import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

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
molecularModel_t1 = SklearnModel(dataset=dataset, model=model, preprocess_x=None)
molecularModel_t1 = SklearnModel(
    dataset=dataset, model=model, preprocess_x=VarianceThreshold()
)
molecularModel_t1 = SklearnModel(
    dataset=dataset, model=model, preprocess_y=VarianceThreshold()
)

molecularModel_t1 = SklearnModel(
    dataset=dataset, model=model, preprocess_x=StandardScaler()
)


preprocess_x = [StandardScaler(), MinMaxScaler()]
pipeline = sklearn.pipeline.Pipeline(steps=[])
for preprocessor in preprocess_x:
    pipeline.steps.append((str(preprocessor), preprocessor))
