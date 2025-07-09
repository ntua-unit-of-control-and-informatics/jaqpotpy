import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from jaqpotpy.models import SklearnModel
from jaqpotpy.datasets import JaqpotTabularDataset
from jaqpotpy.descriptors import RDKitDescriptors
from jaqpot_api_client.models.model_task import ModelTask

# Define the dataset with SMILES strings, a categorical variable, temperature, and activity values
data = pd.DataFrame(
    {
        "smiles": [
            "CC",
            "CCO",
            "CCC",
            "CCCl",
            "CCBr",
            "COC",
            "CCOCC",
            "CCCO",
            "CCCC",
            "CCCCCC",
        ],
        "cat_col": [
            "high",
            "high",
            "high",
            "high",
            "high",
            "low",
            "low",
            "low",
            "low",
            "low",
        ],
        "temperature": np.random.randint(20, 37, size=10),
        "activity": [80, 81, 81, 84, 83.5, 83, 89, 90, 91, 97],
    }
)

# Initialize RDKitDescriptors to calculate molecular descriptors from SMILES
featurizer = RDKitDescriptors()

# Create a JaqpotpyDataset object, specifying features, target, and SMILES columns, and set the task type to regression
train_dataset = JaqpotTabularDataset(
    df=data,
    x_cols=["cat_col", "temperature"],  # Feature columns
    y_cols=["activity"],  # Target column
    smiles_cols=["smiles"],  # SMILES column for molecular descriptor generation
    task=ModelTask.REGRESSION,  # Type of task: regression
    featurizers=featurizer,  # Use RDKit featurizer for SMILES-based features
)

model = RandomForestRegressor(random_state=42)

# Define a preprocessing pipeline for the feature columns
# - First, apply OneHotEncoder to the categorical column 'cat_col'
# - Then, StandardScaler scales numeric data to standardize them
double_preprocessing = [
    ColumnTransformer(
        transformers=[
            ("OneHotEncoder", OneHotEncoder(), ["cat_col"]),
        ],
        remainder="passthrough",  # Keep other columns as they are
        force_int_remainder_cols=False,  # Ensures all columns are compatible for further processing
    ),
    StandardScaler(),  # Standard scaling for numerical features after encoding
]

# Define preprocessing for the target column
single_preprocessing = MinMaxScaler()

jaqpot_model = SklearnModel(
    dataset=train_dataset,
    model=model,
    preprocess_x=double_preprocessing,  # Preprocessing for features
    preprocess_y=single_preprocessing,  # Preprocessing for the target
)
jaqpot_model.fit()


X_test = pd.DataFrame(
    {
        "smiles": ["CCCOC", "CO"],
        "cat_col": ["low", "low"],
        "temperature": [27.0, 22.0],
        "activity": [89.0, 86.0],
    }
)

# Initialize a JaqpotpyDataset for prediction, specifying features and target columns
test_dataset = JaqpotTabularDataset(
    df=X_test,
    smiles_cols="smiles",
    x_cols=["cat_col", "temperature"],
    y_cols=None,
    task=ModelTask.REGRESSION,
    featurizers=featurizer,  # Same featurizer for consistency with training
)


predictions = jaqpot_model.predict(test_dataset)
print(predictions)
