import pandas as pd
import copy
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors import MordredDescriptors
from sklearn.ensemble import RandomForestRegressor
from jaqpotpy.models import SklearnModel
from jaqpotpy.doa import Leverage
from kennard_stone import train_test_split as kennard_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, Binarizer
from sklearn.compose import ColumnTransformer
from jaqpotpy import Jaqpot

pd_df = pd.read_csv("jaqpot_dataset.csv").iloc[:20, :]
bool_columns = pd_df.select_dtypes(include="bool").columns.tolist()
for col in bool_columns:
    pd_df[col] = pd_df[col].astype(str)

x = pd_df.drop(columns=["log(BCF) (L/Kg)"])
y = pd_df["log(BCF) (L/Kg)"]
x_cols = pd_df.drop(
    columns=["Smiles", "Exposure (days)", "Depuration (days)", "log(BCF) (L/Kg)"]
).columns.tolist()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

train_dataset = pd.concat([x_train, y_train], axis=1)
test_dataset = pd.concat([x_test, y_test], axis=1)


train_jaqpot_dataset = JaqpotpyDataset(
    df=train_dataset,
    smiles_cols=["Smiles"],
    x_cols=x_cols,
    y_cols="log(BCF) (L/Kg)",
    task="regression",
    featurizer=MordredDescriptors(),
)

test_jaqpot_dataset = JaqpotpyDataset(
    df=test_dataset,
    smiles_cols=["Smiles"],
    x_cols=pd_df.drop(
        columns=["Smiles", "Exposure (days)", "Depuration (days)", "log(BCF) (L/Kg)"]
    ).columns.tolist(),
    y_cols="log(BCF) (L/Kg)",
    task="regression",
    featurizer=MordredDescriptors(),
)

selected_features = [
    "Species",
    "Sex",
    "Tissue",
    "Mixture",
    "ETA_psi_1",
    "ATSC7i",
    "AATS6dv",
    "ATSC4dv",
    "AATSC6dv",
    "AATSC7d",
    "AATS7dv",
    "GATS8c",
    "nBase",
    "GATS8s",
    "ETA_beta_ns_d",
    "AATSC3s",
    "ATSC0Z",
    "MINsssPbH",
    "MATS1se",
    # "MDEC-33",
    "ATSC2c",
    "ATSC2se",
    "Exposure_Concentration_(ug/L)",
    "ATS8i",
    "AATS8dv",
]

train_jaqpot_dataset.select_features(SelectColumns=selected_features)
test_jaqpot_dataset.select_features(SelectColumns=selected_features)

proprocessing_steps_x = ColumnTransformer(
    transformers=[
        ("OneHotEncoder", OneHotEncoder(), ["Species", "Sex", "Tissue", "Mixture"]),
        # ("Binarizer", Binarizer(threshold=0.5), ["Mixture"]),
    ],
    remainder="passthrough",
    force_int_remainder_cols=False,
)
jaqpot_model = SklearnModel(
    dataset=train_jaqpot_dataset,
    model=RandomForestRegressor(),
    preprocess_x=proprocessing_steps_x,
    doa=Leverage(),
)


jaqpot_model.fit()
jaqpot_model.cross_validate(train_jaqpot_dataset, n_splits=5)
jaqpot_model.evaluate(test_jaqpot_dataset)
jaqpot_model.randomization_test(
    train_dataset=train_jaqpot_dataset, test_dataset=test_jaqpot_dataset, n_iters=10
)

# Upload Model on Jaqpot
jaqpot = Jaqpot()
jaqpot.login()
jaqpot_model.deploy_on_jaqpot(
    jaqpot=jaqpot,
    name="Log BCF PFAS Model",
    description="BCF SCENARIOS MODEL",
    visibility="PUBLIC",
)
