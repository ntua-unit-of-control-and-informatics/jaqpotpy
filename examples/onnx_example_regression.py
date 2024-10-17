import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from jaqpotpy.descriptors.molecular import TopologicalFingerprint, MordredDescriptors
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.models import SklearnModel
from jaqpotpy.doa.doa import Leverage
from jaqpotpy.models.preprocessing import Preprocess
from jaqpotpy import Jaqpot

path = "./jaqpotpy/test_data/test_data_smiles_regression.csv"

df = pd.read_csv(path).iloc[0:100, :]
smiles_cols = ["SMILES"]
y_cols = ["ACTIVITY"]
x_cols = ["X1", "X2"]
featurizer = TopologicalFingerprint()
dataset = JaqpotpyDataset(
    df=df,
    y_cols=y_cols,
    smiles_cols=smiles_cols,
    x_cols=x_cols,
    task="regression",
    featurizer=featurizer,
)
pre = Preprocess()
# pre.register_preprocess_class("Standard Scaler", StandardScaler())
# pre.register_preprocess_class_y("minmax_y", MinMaxScaler())

model = RandomForestRegressor(random_state=42)
doa_method = Leverage()
molecularModel_t1 = SklearnModel(
    dataset=dataset, doa=None, model=model, preprocessor=None, cv=None
)

molecularModel_t1.fit()
# # # print(molecularModel_t1.transformers_y)
# pred_path = "./jaqpotpy/test_data/test_data_smiles_prediction_dataset.csv"
# df = pd.read_csv(pred_path)

# # # # df = pd.DataFrame({
# # # #     'SMILES': ['CC'],
# # # #     'X1': [1],
# # # #     'X2': [2]
# # # })
prediction_dataset = JaqpotpyDataset(
    df=df,
    y_cols=None,
    smiles_cols=smiles_cols,
    x_cols=x_cols,
    task="regression",
    featurizer=featurizer,
)
# # # print(prediction_dataset.df)

# skl_predictions = molecularModel_t1.predict(prediction_dataset)
# # skl_probabilities = molecularModel_t1.predict_proba(prediction_dataset)
