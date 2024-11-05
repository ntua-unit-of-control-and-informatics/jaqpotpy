# Import necessary libraries for data handling, modeling, and feature preprocessing
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from jaqpotpy.models import SklearnModel
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors import RDKitDescriptors

# Define the dataset with SMILES strings, a categorical variable, temperature, and activity values
data = {
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
    "temperature": np.random.randint(
        20, 37, size=10
    ),  # Random integer temperatures from 20 to 36
    "activity": [80, 81, 81, 84, 83.5, 83, 89, 90, 91, 97],  # Defined activity levels
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Initialize RDKitDescriptors to calculate molecular descriptors from SMILES
featurizer = RDKitDescriptors()

# Create a JaqpotpyDataset object, specifying features, target, and SMILES columns, and set the task type to regression
train_dataset = JaqpotpyDataset(
    df=df,
    x_cols=["cat_col", "temperature"],  # Feature columns: categorical and temperature
    y_cols=["activity"],  # Target column: activity
    smiles_cols=["smiles"],  # SMILES column for molecular descriptors
    task="regression",  # Type of task: regression
    featurizer=featurizer,  # Use RDKit featurizer for SMILES-based features
)

# Define the regression model, using RandomForestRegressor from scikit-learn
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
single_preprocessing = MinMaxScaler()  # Scales target values to be between 0 and 1

# Initialize the model with dataset, model, and preprocessing steps
jaqpot_model = SklearnModel(
    dataset=train_dataset,
    model=model,
    preprocess_x=double_preprocessing,  # Preprocessing for features
    preprocess_y=single_preprocessing,  # Preprocessing for the target
)

# Fit the model to the training data
jaqpot_model.fit()

# Define a new test dataset for making predictions
# Includes two samples with SMILES, categorical variable, temperature, and activity
X_test = {
    "smiles": ["CCCOC", "CO"],
    "cat_col": ["low", "low"],
    "temperature": [27.0, 22.0],
    "activity": [89.0, 86.0],
}

# Convert the test data into a DataFrame
df_test = pd.DataFrame(X_test)

# Initialize a JaqpotpyDataset for prediction, specifying features and target columns
test_dataset = JaqpotpyDataset(
    df=df_test,
    smiles_cols="smiles",  # Column with SMILES strings for molecular features
    x_cols=["cat_col", "temperature"],  # Feature columns
    y_cols=None,
    task="regression",
    featurizer=featurizer,  # Same featurizer for consistency with training
)


# Use the trained model to predict activity values for the test dataset
predictions = jaqpot_model.predict(test_dataset)

# Print the predicted activity values
print(predictions)
