import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from jaqpotpy.descriptors.molecular import TopologicalFingerprint, MordredDescriptors
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.models import SklearnModel

# Regression example
path = "./jaqpotpy/test_data/test_data_smiles_regression.csv"

df = pd.read_csv(path)
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


model = RandomForestRegressor(random_state=42)
TestModel = SklearnModel(dataset=dataset, doa=None, model=model)

TestModel.fit()

path = "./jaqpotpy/test_data/test_data_smiles_prediction_dataset.csv"
df2 = pd.read_csv(path)
smiles_cols = ["SMILES"]
y_cols = ["ACTIVITY"]
x_cols = ["X1", "X2"]
featurizer = TopologicalFingerprint()
prediction_dataset = JaqpotpyDataset(
    df=df,
    y_cols=y_cols,
    smiles_cols=smiles_cols,
    x_cols=x_cols,
    task="regression",
    featurizer=featurizer,
)
x = TestModel.evaluate(prediction_dataset)
y = TestModel.cross_validate(dataset, n_splits=2)


# Classification example
path_multi_class = "./jaqpotpy/test_data/test_data_smiles_multi_classification.csv"

multi_classification_df = pd.read_csv(path_multi_class)

featurizer = TopologicalFingerprint()
dataset_multi_class = JaqpotpyDataset(
    df=multi_classification_df,
    y_cols=["ACTIVITY"],
    smiles_cols=["SMILES"],
    x_cols=["X1", "X2"],
    task="multiclass_classification",
    featurizer=featurizer,
)
model = RandomForestClassifier(random_state=42)
pre = StandardScaler()
jaqpot_model = SklearnModel(
    dataset=dataset_multi_class, doa=None, model=model, preprocess_x=pre
)
jaqpot_model.fit(onnx_options={StandardScaler: {"div": "div_cast"}})
jaqpot_model.evaluate(dataset_multi_class)
jaqpot_model.cross_validate(dataset_multi_class, n_splits=2)


# Multi-output regression
path_multi_reg = "./jaqpotpy/test_data/test_data_smiles_regression_multioutput.csv"
path_multi_reg_df = pd.read_csv(path_multi_reg)
featurizer = TopologicalFingerprint()
dataset_multi_reg = JaqpotpyDataset(
    df=path_multi_reg_df,
    y_cols=["ACTIVITY", "ACTIVITY_2"],
    smiles_cols=["SMILES"],
    x_cols=["X1", "X2"],
    task="regression",
    featurizer=featurizer,
)
pre = StandardScaler()

model = RandomForestRegressor(random_state=42)
jaqpot_model = SklearnModel(
    dataset=dataset_multi_reg, doa=None, model=model, preprocess_x=pre
)
jaqpot_model.fit()


jaqpot_model.evaluate(dataset_multi_reg)
jaqpot_model.cross_validate(dataset_multi_reg, n_splits=2)
