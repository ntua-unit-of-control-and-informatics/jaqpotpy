import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.models import SklearnModel
from jaqpotpy.doa.doa import Leverage
from jaqpotpy.models.preprocessing import Preprocess
from jaqpotpy import Jaqpot

path = "jaqpotpy/test_data/test_data_smiles_CATEGORICAL_classification.csv"
df = pd.read_csv(path)
df = df.drop(columns=["SMILES"])
smiles_cols = None  # ["SMILES"]
y_cols = ["ACTIVITY"]
x_cols = ["X1", "X2", "Cat_col"]
featurizer = None  # MACCSKeysFingerprint()

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
        ("Standard Scaler", StandardScaler(), ["X1", "X2"]),
        ("OneHotEncoder", OneHotEncoder(), ["Cat_col"]),
    ],
    remainder="passthrough",
)

pre = Preprocess()
pre.register_preprocess_class("ColumnTransformer", column_transormer)

model = RandomForestClassifier(random_state=42)
molecularModel_t1 = SklearnModel(
    dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=pre
)
molecularModel_t1.fit()

prediction_dataset = JaqpotpyDataset(
    df=df.iloc[0:5, :],
    y_cols=None,
    smiles_cols=smiles_cols,
    x_cols=x_cols,
    task="regression",
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
