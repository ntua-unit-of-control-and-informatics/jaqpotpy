import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold

from jaqpotpy import Jaqpot
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors.molecular import (
    MACCSKeysFingerprint,
)
from jaqpotpy.doa.doa import Leverage
from jaqpotpy.models import SklearnModel
from jaqpotpy.models.preprocessing import Preprocess

path = "../jaqpotpy/test_data/test_data_smiles_classification.csv"

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

sel = VarianceThreshold(threshold=0.1)
dataset.select_features(sel)
print(dataset.X.shape)
pre = Preprocess()
# pre.register_preprocess_class("Standard Scaler", StandardScaler())
# pre.register_preprocess_class_y("minmax_y", MinMaxScaler())

model = RandomForestRegressor(random_state=42)
doa_method = Leverage()
molecularModel_t1 = SklearnModel(
    dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=None
)

molecularModel_t1.fit()
# pred_path = "./jaqpotpy/test_data/test_data_smiles_prediction_dataset.csv"
# df = pd.read_csv(pred_path)


# skl_predictions = molecularModel_t1.predict(prediction_dataset)
# skl_probabilities = molecularModel_t1.predict_proba(prediction_dataset)
# onnx_predictions = molecularModel_t1.predict_onnx(prediction_dataset)
# onnx_probs = molecularModel_t1.predict_proba_onnx(prediction_dataset)
# print("SKLearn Predictions:", skl_predictions)
# print('SKLearn Probabilities:', skl_probabilities)
# print("ONNX Predictions:", onnx_predictions)
# # print('ONNX Probabilities:', onnx_probs)


# Upload locally
# jaqpot = Jaqpot(
#     base_url="http://localhost.jaqpot.org",
#     app_url="http://localhost.jaqpot.org:3000",
#     login_url="http://localhost.jaqpot.org:8070",
#     api_url="http://localhost.jaqpot.org:8080",
#     keycloak_realm="jaqpot-local",
#     keycloak_client_id="jaqpot-local-test",
# )

jaqpot = Jaqpot()
jaqpot.login()
molecularModel_t1.deploy_on_jaqpot(
    jaqpot=jaqpot,
    name="Demo: Regression Maccs Keys",
    description="Test new api files",
    visibility="PRIVATE",
)
