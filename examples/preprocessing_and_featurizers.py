import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
)
from sklearn.compose import ColumnTransformer
from jaqpotpy.models import SklearnModel
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors import RDKitDescriptors


data = {
    "smiles": [
        "CCO",
        "CCN",
        "CCC",
        "CCCl",
        "CCBr",
        "COC",
        "CCOCC",
        "CCCO",
        "CCCC",
        "CCCS",
    ],
    "cat_col": [
        "high",
        "low",
        "high",
        "low",
        "high",
        "low",
        "high",
        "low",
        "high",
        "low",
    ],
    "temperature": np.linspace(20, 37, 10),
    "activity": np.random.randint(1, 101, size=10),
}

# Create the DataFrame
df = pd.DataFrame(data)
featurizer = RDKitDescriptors()

# Step 3: Initialize a JaqpotpyDataset with the DataFrame
# Specify the feature columns and the target column, and define the task as regression.
dataset = JaqpotpyDataset(
    df=df,
    x_cols=["category", "temperature"],
    y_cols=["activity"],
    smiles_cols=["smiles"],
    task="regression",  # select among "regression", "binary_classification" and "multiclass_classification"
    featurizer=featurizer,
)

model = RandomForestRegressor(random_state=42)
double_preprocessing = [
    ColumnTransformer(
        transformers=[
            ("OneHotEncoder", OneHotEncoder(), ["cat_col"]),
        ],
        remainder="passthrough",
    ),
    StandardScaler(),
]
single_preprocessing = MinMaxScaler()
jaqpot_model = SklearnModel(
    dataset=dataset,
    model=model,
    preprocess_x=double_preprocessing,
    preprocess_y=single_preprocessing,
)
jaqpot_model.fit()


X_test = np.array(
    [
        [
            "CCC",
            "COC",
        ],
        ["low", "low"],
        ["27", "22"],
    ]
)


df_test = pd.DataFrame(X_test, columns=["smiles", "cat_col", "temperature"])
test_dataset = JaqpotpyDataset(
    df=df_test,
    smiles_cols="smiles",
    x_cols=["cat_col", "temperature"],
    y_cols=None,
    task="regression",
    featurizer=featurizer,
)

predictions = jaqpot_model.predict(test_dataset)

print(predictions)
