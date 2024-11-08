import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from jaqpotpy.descriptors.molecular import TopologicalFingerprint, MordredDescriptors
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.models import SklearnModel
from jaqpotpy.doa import Leverage, BoundingBox, MeanVar
from jaqpotpy import Jaqpot

path = (
    "../jaqpotpy/test_data/test_data_smiles_CATEGORICAL_classification_LABELS_new.csv"
)

df = pd.read_csv(path).iloc[0:100, :]
smiles_cols = ["SMILES"]
y_cols = ["ACTIVITY"]
x_cols = ["X1", "X2", "Cat_col"]
featurizer = TopologicalFingerprint()
dataset = JaqpotpyDataset(
    df=df,
    y_cols=y_cols,
    smiles_cols=smiles_cols,
    x_cols=x_cols,
    task="BINARY_CLASSIFICATION",
    featurizer=featurizer,
)

model = RandomForestClassifier(random_state=42)
preprocess_x = ColumnTransformer(
    transformers=[
        ("OneHotEncoder", OneHotEncoder(), ["Cat_col"]),
    ],
    remainder="passthrough",
)

molecularModel_t1 = SklearnModel(
    dataset=dataset,
    doa=BoundingBox(),
    model=model,
    preprocess_x=preprocess_x,
    preprocess_y=LabelEncoder(),
)

molecularModel_t1.fit()
pred_path = "../jaqpotpy/test_data/test_data_smiles_categorical_prediction_dataset.csv"
test_dataset = pd.read_csv(pred_path)

prediction_dataset = JaqpotpyDataset(
    df=test_dataset,
    smiles_cols=smiles_cols,
    x_cols=x_cols,
    y_cols=y_cols,
    task="BINARY_CLASSIFICATION",
    featurizer=featurizer,
)
# molecularModel_t1.evaluate(dataset)
# print(molecularModel_t1.train_scores)
# print(molecularModel_t1.test_scores)
print("TEST", molecularModel_t1.evaluate(dataset=prediction_dataset))
evaluation_metrics = molecularModel_t1.evaluate(dataset=prediction_dataset)

# skl_predictions = molecularModel_t1.predict(prediction_dataset)
# skl_probabilities = molecularModel_t1.predict_proba(prediction_dataset)
# onnx_predictions = molecularModel_t1.predict_onnx(prediction_dataset)
# onnx_probs = molecularModel_t1.predict_proba_onnx(prediction_dataset)
# print("SKLearn Predictions:", skl_predictions)
# print("SKLearn Probabilities:", skl_probabilities)
# print("ONNX Predictions:", onnx_predictions)
# print("ONNX Probabilities:", onnx_probs)

# Upload locally
jaqpot = Jaqpot(
    base_url="http://localhost.jaqpot.org",
    app_url="http://localhost.jaqpot.org:3000",
    login_url="http://localhost.jaqpot.org:8070",
    api_url="http://localhost.jaqpot.org:8080",
    keycloak_realm="jaqpot-local",
    keycloak_client_id="jaqpot-local-test",
)

# jaqpot = Jaqpot()
jaqpot.login()
molecularModel_t1.deploy_on_jaqpot(
    jaqpot=jaqpot,
    name="Demo: Regression topological and minmax scaler on y",
    description="Test",
    visibility="PRIVATE",
)
