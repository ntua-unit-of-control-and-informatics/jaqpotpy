import pandas as pd

from jaqpotpy import Jaqpot
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors.molecular import (
    MACCSKeysFingerprint,
)

# from jaqpotpy.models import SklearnModel
from jaqpotpy.models import XGBoostModel
from xgboost import XGBClassifier

path = "jaqpotpy/test_data/test_data_smiles_classification.csv"

df_train = pd.read_csv(path).iloc[0:80, :]
df_predict = pd.read_csv(path).iloc[80:100, :]
smiles_cols = ["SMILES"]
y_cols = ["ACTIVITY"]
featurizer = MACCSKeysFingerprint()
dataset = JaqpotpyDataset(
    df=df_train,
    y_cols=y_cols,
    smiles_cols=smiles_cols,
    task="binary_classification",
    featurizer=featurizer,
)

model = XGBClassifier(random_state=42)
molecularModel_t1 = XGBoostModel(dataset=dataset, doa=None, model=model)

molecularModel_t1.fit()
prediction_dataset = JaqpotpyDataset(
    df=df_predict,
    y_cols=y_cols,
    smiles_cols=smiles_cols,
    task="binary_classification",
    featurizer=featurizer,
)

skl_predictions = molecularModel_t1.predict(prediction_dataset)
skl_probabilities = molecularModel_t1.predict_proba(prediction_dataset)
onnx_predictions = molecularModel_t1.predict_onnx(prediction_dataset)
onnx_probs = molecularModel_t1.predict_proba_onnx(prediction_dataset)
print("SKLearn Predictions:", skl_predictions)
print("SKLearn Probabilities:", skl_probabilities)
print("ONNX Predictions:", onnx_predictions)
print("ONNX Probabilities:", onnx_probs)


# Upload locally
jaqpot = Jaqpot(
    base_url="http://localhost.jaqpot.org",
    app_url="http://localhost.jaqpot.org:3000",
    login_url="http://localhost.jaqpot.org:8070",
    api_url="http://localhost.jaqpot.org:8080",
    keycloak_realm="jaqpot-local",
    keycloak_client_id="jaqpot-local-test",
)

jaqpot.login()
molecularModel_t1.deploy_on_jaqpot(
    jaqpot=jaqpot,
    name="Demo: XGBOOST model seperate",
    description="Compare with XGBOOST",
    visibility="PRIVATE",
)
