import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from jaqpotpy import Jaqpot
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors.molecular import MACCSKeysFingerprint, TopologicalFingerprint
from jaqpotpy.doa import Leverage, BoundingBox, MeanVar
from jaqpotpy.models import SklearnModel

path = "jaqpotpy/test_data/test_data_smiles_regression.csv"

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

# sel = VarianceThreshold(threshold=0.1)
# dataset.select_features(sel)
# print(dataset.X.shape)

model = RandomForestRegressor(random_state=42)
doa_methods = [Leverage(), BoundingBox(), MeanVar()]
x_preprocessor = StandardScaler()
molecularModel_t1 = SklearnModel(
    dataset=dataset,
    doa=doa_methods,
    model=model,
    preprocess_x=x_preprocessor,
)

molecularModel_t1.fit(onnx_options={StandardScaler: {"div": "div_cast"}})
pred_path = "jaqpotpy/test_data/test_data_smiles_prediction_dataset.csv"
pred_df = pd.read_csv(pred_path)
prediction_dataset = JaqpotpyDataset(
    df=pred_df,
    y_cols=None,
    smiles_cols=smiles_cols,
    x_cols=x_cols,
    task="regression",
    featurizer=featurizer,
)

skl_predictions = molecularModel_t1.predict(prediction_dataset)
# # skl_probabilities = molecularModel_t1.predict_proba(prediction_dataset)
onnx_predictions = molecularModel_t1.predict_onnx(prediction_dataset)
# # onnx_probs = molecularModel_t1.predict_proba_onnx(prediction_dataset)
print("SKLearn Predictions:", skl_predictions)
# # print('SKLearn Probabilities:', skl_probabilities)
print("ONNX Predictions:", onnx_predictions)
# # # print('ONNX Probabilities:', onnx_probs)


# Upload locally
# jaqpot = Jaqpot(
#     base_url="http://localhost.jaqpot.org",
#     app_url="http://localhost.jaqpot.org:3000",
#     login_url="http://localhost.jaqpot.org:8070",
#     api_url="http://localhost.jaqpot.org:8080",
#     keycloak_realm="jaqpot-local",
#     keycloak_client_id="jaqpot-local-test",
# )

# # jaqpot = Jaqpot()
# jaqpot.login()
# molecularModel_t1.deploy_on_jaqpot(
#     jaqpot=jaqpot,
#     name="Demo: Regression topological and minmax scaler on y",
#     description="Test",
#     visibility="PRIVATE",
# )
