# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from jaqpotpy.models import SklearnModel
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors import RDKitDescriptors

# Create sample data
data = {
    "smiles": [  # List of SMILES strings representing molecular structures
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
    "cat_col": [  # Categorical feature column with two levels, "high" and "low"
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
    "temperature": np.random.randint(20, 37, size=10),  # Random temperature values
    "activity": [
        80,
        81,
        81,
        84,
        83.5,
        83,
        89,
        90,
        91,
        97,
    ],  # Target variable 'activity'
}

# Create DataFrame from data
df = pd.DataFrame(data)

# Initialize a molecular featurizer to generate molecular descriptors from SMILES
featurizer = RDKitDescriptors()

# Prepare the dataset for training with Jaqpotpy
train_dataset = JaqpotpyDataset(
    df=df,
    x_cols=[
        "cat_col",
        "temperature",
    ],  # Feature columns including categorical and temperature
    y_cols=["activity"],  # Target variable column
    smiles_cols=["smiles"],  # Column containing SMILES strings for featurization
    task="regression",  # Task type is regression
    featurizer=featurizer,  # Featurizer to apply to the 'smiles' column
)

# Initialize the machine learning model
model = RandomForestRegressor(random_state=42)

# Wrap the model and dataset in a SklearnModel object
jaqpot_model = SklearnModel(dataset=train_dataset, model=model)

# Train the model on the dataset
jaqpot_model.fit()

# Perform cross-validation on the training data to estimate model performance
# cross_validate function divides the data into 'n_splits' (e.g., 10) folds, training on 'n-1' folds and testing on the remaining fold.
# This process is repeated to ensure the model's ability to generalize to unseen data. Note that the same JaqpotpyDataset that was provided
# during model fitting should be provided to the cross_validate() method.
jaqpot_model.cross_validate(train_dataset, n_splits=10)

# Define test data for external evaluation
X_test = {
    "smiles": ["CCCOC", "CO"],  # New SMILES strings for prediction
    "cat_col": ["low", "low"],  # Categorical feature values for test data
    "temperature": [27.0, 22.0],  # Temperature values for test data
    "activity": [
        89.0,
        86.0,
    ],  # Target activity values for reference (not used for prediction)
}

# Create DataFrame for test data
df_test = pd.DataFrame(X_test)

# Prepare the test dataset with Jaqpotpy
test_dataset = JaqpotpyDataset(
    df=df_test,
    smiles_cols="smiles",
    x_cols=["cat_col", "temperature"],
    y_cols=["activity"],
    task="regression",
    featurizer=featurizer,
)

# Evaluate the model on the test dataset
# The evaluate function uses the test dataset to assess the model's performance on new/unseen data,
# providing metrics that reflect its predictive ability outside of training data.
jaqpot_model.evaluate(test_dataset)

# Generate predictions for the test data
predictions = jaqpot_model.predict(test_dataset)

# Print predictions to view model output on test data
print(predictions)
