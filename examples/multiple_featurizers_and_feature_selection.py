# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from jaqpotpy.models import SklearnModel
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors import RDKitDescriptors, MACCSKeysFingerprint

# Sample data creation
data = {
    "smiles": [  # List of SMILES strings for molecular structures
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
    "cat_col": [  # Categorical column with two levels, "high" and "low"
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
    ),  # Random temperatures between 20 and 37
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

# Create DataFrame from the sample data
df = pd.DataFrame(data)

# At this multiple molecular featurizers from the list of available featurizers can be provided to JaqpotpyDataset as a list
featurizers = [RDKitDescriptors(), MACCSKeysFingerprint()]

# Create a JaqpotpyDataset object for training, specifying data columns and task type
train_dataset = JaqpotpyDataset(
    df=df,
    x_cols=[
        "cat_col",
        "temperature",
    ],  # Features include categorical and temperature columns
    y_cols=["activity"],  # Target column
    smiles_cols=["smiles"],  # Column containing SMILES strings for featurization
    task="regression",  # Task type is regression
    featurizer=featurizers,  # List of featurizers to apply to 'smiles'
)

# A lot of features are now included in the dataset, so we strongly recommend following a feature selection process.
# All skleanr feature selector methods are supported. Note that categorical variables cannot be included in the selection pipeline,
# so they should be excluded by explicitly including them in the ExcludeColumns argument.
FeatureSelector = VarianceThreshold(threshold=0.1)
train_dataset.select_features(
    FeatureSelector,
    ExcludeColumns=["cat_col"],  # Explicitly exclude categorical variables
)

# Alternative feature selection method: directly select specific columns by name and provide them through the SelectColumns argument
myList = [
    "temperature",
    "MaxAbsEStateIndex",
    "MaxEStateIndex",
    "MinAbsEStateIndex",
    "MinEStateIndex",
    "SPS",
    "MolWt",
    "HeavyAtomMolWt",
]
train_dataset.select_features(SelectColumns=myList)

# Initialize the model with a RandomForestRegressor
model = RandomForestRegressor(random_state=42)

# Wrap the model and dataset in a SklearnModel object for easy training and prediction with jaqpotpy
jaqpot_model = SklearnModel(dataset=train_dataset, model=model)
jaqpot_model.fit()  # Fit the model to the training data

# Define test data with new molecular structures and features for prediction
X_test = {
    "smiles": ["CCCOC", "CO"],  # New SMILES strings for testing
    "cat_col": ["low", "low"],  # Categorical values for test samples
    "temperature": [27.0, 22.0],  # Temperature values for test samples
}

# Create a DataFrame for test data
df_test = pd.DataFrame(X_test)

# Create a JaqpotpyDataset object for testing, using same column setup as training dataset
test_dataset = JaqpotpyDataset(
    df=df_test,
    smiles_cols="smiles",
    x_cols=["cat_col", "temperature"],
    y_cols=None,
    task="regression",
    featurizer=featurizers,
)

predictions = jaqpot_model.predict(test_dataset)
print(predictions)
