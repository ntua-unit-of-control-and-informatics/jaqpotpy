import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from jaqpotpy.descriptors.molecular import (
    TopologicalFingerprint,
    MordredDescriptors,
    MACCSKeysFingerprint,
)
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.models import SklearnModel
from jaqpotpy.doa.doa import Leverage
from jaqpotpy.models.preprocessing import Preprocess
from jaqpotpy import Jaqpot

path = "./jaqpotpy/test_data/test_data_smiles_regression.csv"

df = pd.read_csv(path).iloc[0:3, :]
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
# print(featurizer.__dict__)

# attributes = copy.deepcopy(featurizer.__dict__)
# print(attributes)
pre = Preprocess()
# pre.register_preprocess_class("Standard Scaler", StandardScaler())
# pre.register_preprocess_class_y("minmax_y", MinMaxScaler())

model = RandomForestRegressor(random_state=42)
doa_method = Leverage()
molecularModel_t1 = SklearnModel(
    dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=None
)

molecularModel_t1.fit()
# # # print(molecularModel_t1.transformers_y)
# pred_path = "./jaqpotpy/test_data/test_data_smiles_prediction_dataset.csv"
pred_path = "/Users/vassilis/Desktop/test_csv.csv"
df = pd.read_csv(pred_path)

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
# # print(prediction_dataset.df)

skl_predictions = molecularModel_t1.predict(prediction_dataset)
# skl_probabilities = molecularModel_t1.predict_proba(prediction_dataset)
onnx_predictions = molecularModel_t1.predict_onnx(prediction_dataset)
# onnx_probs = molecularModel_t1.predict_proba_onnx(prediction_dataset)
print("SKLearn Predictions:", skl_predictions)
# print('SKLearn Probabilities:', skl_probabilities)
print("ONNX Predictions:", onnx_predictions)
# print('ONNX Probabilities:', onnx_probs)


# Merge predictions and probabilities into a pandas DataFrame
df_predictions = pd.DataFrame(
    {
        "SKLearn Predictions": skl_predictions,
        # 'SKLearn Probabilities': skl_probabilities,
        "ONNX Predictions": onnx_predictions,
        # 'ONNX Probabilities': onnx_probs
    }
)
print(df_predictions)


# # # Upload locally
# jaqpot = Jaqpot(
#     base_url="http://localhost.jaqpot.org",
#     app_url="http://localhost.jaqpot.org:3000",
#     login_url="http://localhost.jaqpot.org:8070",
#     api_url="http://localhost.jaqpot.org:8080",
#     keycloak_realm="jaqpot-local",
#     keycloak_client_id="jaqpot-local-test",
# )

jaqpot = Jaqpot(
    base_url="http://localhost.jaqpot.org",
    app_url="http://localhost.jaqpot.org:3000",
    login_url="http://localhost.jaqpot.org:8070",
    api_url="http://localhost.jaqpot.org:8080",
)
with open("/Users/vassilis/Desktop/api_key.txt", "r") as file:
    api_key = file.read().strip()
jaqpot.set_api_key(api_key)
# jaqpot.login()
molecularModel_t1.deploy_on_jaqpot(
    jaqpot=jaqpot,
    name="Demo: Regression Maccs Keys",
    description="Test new api files",
    visibility="PRIVATE",
)
