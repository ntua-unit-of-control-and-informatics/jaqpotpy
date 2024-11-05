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
from jaqpotpy.doa import Leverage
from jaqpotpy import Jaqpot

path = "jaqpotpy/test_data/test_data_smiles_CATEGORICAL_classification_LABELS_new.csv"
df = pd.read_csv(path)

# df = df.drop(columns=["SMILES"])
smiles_cols = ["SMILES"]
y_cols = ["ACTIVITY"]
x_cols = ["X1", "X2", "Cat_col"]

featurizer = TopologicalFingerprint(size=5)

dataset = JaqpotpyDataset(
    df=df.iloc[0:100, :],
    y_cols=y_cols,
    smiles_cols=smiles_cols,
    x_cols=x_cols,
    task="BINARY_CLASSIFICATION",
    featurizer=featurizer,
)
preprocess_x = ColumnTransformer(
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
    preprocess_x=preprocess_x,
    preprocess_y=[LabelEncoder()],
)
molecularModel_t1.fit()
pred_path = "jaqpotpy/test_data/test_data_smiles_categorical_prediction_dataset.csv"
df = pd.read_csv(pred_path)
prediction_dataset = JaqpotpyDataset(
    df=df,
    y_cols=y_cols,
    smiles_cols=smiles_cols,
    x_cols=x_cols,
    task="BINARY_CLASSIFICATION",
    featurizer=featurizer,
)

# train_scores = molecularModel_t1.train_scores
# print(train_scores["confusionMatrix"])
# evaluation_scores = molecularModel_t1.evaluate(prediction_dataset)
# print(evaluation_scores["accuracy"])
# print(evaluation_scores["confusionMatrix"])
molecularModel_t1.cross_validate(dataset, n_splits=3)
cross_val_scores = molecularModel_t1.cross_val_scores
print(cross_val_scores["output_0"]["fold_1"]["confusionMatrix"])
print(cross_val_scores["output_0"]["fold_3"]["confusionMatrix"])


# skl_predictions = molecularModel_t1.predict(prediction_dataset)
# # skl_probabilities = molecularModel_t1.predict_proba(prediction_dataset)
# onnx_predictions = molecularModel_t1.predict_onnx(prediction_dataset)
# # onnx_probs = molecularModel_t1.predict_proba_onnx(prediction_dataset)
# print("SKLearn Predictions:", skl_predictions)
# # print("SKLearn Probabilities:", skl_probabilities)
# print("ONNX Predictions:", onnx_predictions)
# # print("ONNX Probabilities:", onnx_probs)

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
#     name="Demo: Classification with String Labels",
#     description="Test",
#     visibility="PRIVATE",
# )
