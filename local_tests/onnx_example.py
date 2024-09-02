import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from jaqpotpy.descriptors.molecular import TopologicalFingerprint
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.models import SklearnModel
from jaqpotpy.doa.doa import Leverage
from jaqpotpy.models.preprocessing import Preprocess

path = "/Users/vassilis/Documents/GitHub/jaqpotpy/jaqpotpy/test_data/test_data_smiles_regression_multioutput.csv"

df = pd.read_csv(path)
smiles_cols = ["SMILES"]
y_cols = ["ACTIVITY", "ACTIVITY_2"]
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
# pre.register_preprocess_class('MinMax Scaler', MinMaxScaler())
pre.register_preprocess_class_y("minmax_y", MinMaxScaler())

model = RandomForestRegressor(random_state=42)
doa_method = Leverage()
molecularModel_t1 = SklearnModel(
    dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=pre
)

molecularModel_t1.fit(onnx_options={StandardScaler: {"div": "div_cast"}})

pred_path = "/Users/vassilis/Documents/GitHub/jaqpotpy/jaqpotpy/test_data/test_data_smiles_prediction_dataset.csv"
df = pd.read_csv(pred_path)

# # data = [{'SMILES': 'CC', 'X1': 1, 'X2': 2}]
# # df = pd.DataFrame(data)
prediction_dataset = JaqpotpyDataset(
    df=df,
    y_cols=None,
    smiles_cols=smiles_cols,
    x_cols=x_cols,
    task="regression",
    featurizer=featurizer,
)

skl_predictions = molecularModel_t1.predict(prediction_dataset)
# skl_probabilities = molecularModel_t1.predict_proba(prediction_dataset)
onnx_predictions = molecularModel_t1.predict_onnx(prediction_dataset)
# onnx_probs = molecularModel_t1.predict_proba_onnx(prediction_dataset)
print("SKLearn Predictions:", skl_predictions)
# print('SKLearn Probabilities:', skl_probabilities)
print("ONNX Predictions:", onnx_predictions)
# print('ONNX Probabilities:', onnx_probs)


# Merge predictions and probabilities into a pandas DataFrame
# df_predictions = pd.DataFrame({
#     'SKLearn Predictions': skl_predictions,
#     # 'SKLearn Probabilities': skl_probabilities,
#     'ONNX Predictions': onnx_predictions,
#     # 'ONNX Probabilities': onnx_probs
# })
# print(df_predictions)


# with open('/Users/vassilis/Desktop/api_key.txt', 'r') as file:
#     api_key = file.read().strip()

# jaqpot = Jaqpot("http://localhost.jaqpot.org:8080")
# # # jaqpot = Jaqpot("https://api.appv2.jaqpot.org")
# # # jaqpot.login('jaqpot','jaqpot')
# jaqpot.set_api_key(api_key)
# molecularModel_t1.deploy_on_jaqpot(jaqpot=jaqpot, name="Classification model", description="Test classification with one output ", visibility="PUBLIC")
