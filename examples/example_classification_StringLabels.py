import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.compose import ColumnTransformer
from jaqpotpy.descriptors.molecular import (
    TopologicalFingerprint,
    MordredDescriptors,
    MACCSKeysFingerprint,
)
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.models import SklearnModel
from jaqpotpy.doa.doa import Leverage
from jaqpotpy import Jaqpot

path = "jaqpotpy/test_data/test_data_smiles_CATEGORICAL_classification_LABELS.csv"
df = pd.read_csv(path)

# df = df.drop(columns=["SMILES"])
smiles_cols = None  # ["SMILES"]
y_cols = ["ACTIVITY"]
x_cols = ["X1", "X2", "Cat_col"]
# x_cols = ["Cat_col", "Cat_col2"]

featurizer = TopologicalFingerprint(size=10)

dataset = JaqpotpyDataset(
    df=df,
    y_cols=y_cols,
    smiles_cols=smiles_cols,
    x_cols=x_cols,
    task="BINARY_CLASSIFICATION",
    featurizer=featurizer,
)
column_transormer = ColumnTransformer(
    transformers=[
        # ("Standard Scaler", StandardScaler(), ["X1", "X2"]),
        ("OneHotEncoder", OneHotEncoder(), ["Cat_col"]),
    ],
    remainder="passthrough",
)

model = RandomForestClassifier(random_state=42)
molecularModel_t1 = SklearnModel(
    dataset=dataset,
    doa=None,
    model=model,
    preprocess_x=column_transormer,
    preprocess_y=[LabelEncoder()],
)
molecularModel_t1.fit()
pred_path = "/Users/vassilis/Desktop/test_ohe.csv"
df = pd.read_csv(pred_path)
prediction_dataset = JaqpotpyDataset(
    df=df,
    y_cols=None,
    smiles_cols=smiles_cols,
    x_cols=x_cols,
    task="BINARY_CLASSIFICATION",
    featurizer=featurizer,
)

# skl_predictions = molecularModel_t1.predict(prediction_dataset)
# skl_probabilities = molecularModel_t1.predict_proba(prediction_dataset)
onnx_predictions = molecularModel_t1.predict_onnx(prediction_dataset)
# onnx_probs = molecularModel_t1.predict_proba_onnx(prediction_dataset)
# print("SKLearn Predictions:", skl_predictions)
# print("SKLearn Probabilities:", skl_probabilities)
print("ONNX Predictions:", onnx_predictions)
# print("ONNX Probabilities:", onnx_probs)

# jaqpot = Jaqpot(
#     base_url="http://localhost.jaqpot.org",
#     app_url="http://localhost.jaqpot.org:3000",
#     login_url="http://localhost.jaqpot.org:8070",
#     api_url="http://localhost.jaqpot.org:8080",
# )
# with open("/Users/vassilis/Desktop/api_key.txt", "r") as file:
#     api_key = file.read().strip()
# jaqpot.set_api_key(api_key)
# # jaqpot.login()
# molecularModel_t1.deploy_on_jaqpot(
#     jaqpot=jaqpot,
#     name="Demo Classification: One Hot Encoding",
#     description="Test OHE",
#     visibility="PRIVATE",
# )
