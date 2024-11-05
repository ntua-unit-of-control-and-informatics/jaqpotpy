import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.models import SklearnModel
from jaqpotpy import Jaqpot
from jaqpotpy.doa import MeanVar, BoundingBox, Leverage


path = "jaqpotpy_data.csv"

# TODO: fix functionality
df = pd.read_csv(path)
df = df.rename(columns={"model_2_values": "cell_viability"})
df = df.drop(columns=["type", "bmd_id"])
X, y = df.drop("cell_viability", axis=1), df["cell_viability"]
X["func"] = X["func"].fillna("no_func")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

x_cols = [
    "Substance",
    "size_class",
    "layer",
    "time",
    "func",
    "cell_type_general",
    "species",
    "assay",
    "media",
    "dose",
]
y_cols = ["cell_viability"]

cat_cols = [
    "Substance",
    "size_class",
    "cell_type_general",
    "species",
    "assay",
    "media",
    "func",
]

train_dataset = JaqpotpyDataset(
    df=train_df,
    y_cols=y_cols,
    x_cols=x_cols,
    task="REGRESSION",
)


column_transormer = ColumnTransformer(
    transformers=[
        (
            "OneHotEncoder",
            OneHotEncoder(),
            cat_cols,
        ),
        (
            "MinMaxScaler",
            MinMaxScaler(),
            ["layer", "time", "dose"],
        ),
    ],
    remainder="passthrough",
)

model = MLPRegressor(
    solver="lbfgs",
    random_state=42,
    early_stopping=True,
    max_iter=5000,
    hidden_layer_sizes=30,
)


doa = [Leverage(), MeanVar(), BoundingBox()]
jaqpotModel = SklearnModel(
    dataset=train_dataset,
    doa=doa,
    model=model,
    preprocess_x=column_transormer,
)
jaqpotModel.fit()

test_dataset = JaqpotpyDataset(
    df=test_df,
    y_cols=y_cols,
    x_cols=x_cols,
    task="REGRESSION",
)


x = jaqpotModel.evaluate(test_dataset)
x = jaqpotModel.cross_validate(train_dataset, n_splits=10)

print(x)

jaqpot = Jaqpot()
jaqpot.login()
jaqpotModel.deploy_on_jaqpot(
    jaqpot=jaqpot,
    name="Test predictive model",
    description="Test",
    visibility="PRIVATE",
)
