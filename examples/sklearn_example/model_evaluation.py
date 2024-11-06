# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from jaqpotpy.models import SklearnModel
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors import RDKitDescriptors

# Create sample data
data = pd.DataFrame(
    {
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
        "temperature": np.random.randint(20, 37, size=10),
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
        ],
    }
)

featurizer = RDKitDescriptors()

# Prepare the dataset for training with Jaqpotpy
train_dataset = JaqpotpyDataset(
    df=data,
    x_cols=[
        "temperature",
    ],
    y_cols=["activity"],
    smiles_cols=["smiles"],
    task="regression",
    featurizer=featurizer,
)

model = RandomForestRegressor(random_state=42)
jaqpot_model = SklearnModel(dataset=train_dataset, model=model)
# Set random seed to have reproducibility of results
jaqpot_model.random_seed = 1231
jaqpot_model.fit()

# Perform cross-validation on the training data to estimate model performance
# cross_validate function divides the data into 'n_splits' (e.g., 10) folds, training on 'n-1' folds and testing on the remaining fold.
# This process is repeated to ensure the model's ability to generalize to unseen data. Note that the same JaqpotpyDataset that was provided
# during model fitting should be provided to the cross_validate() method.
jaqpot_model.cross_validate(train_dataset, n_splits=10)

# Define test data for external evaluation
X_test = pd.DataFrame(
    {
        "smiles": ["CCCOC", "CO"],  # New SMILES strings for prediction
        "cat_col": ["low", "low"],  # Categorical feature values for test data
        "temperature": [27.0, 22.0],  # Temperature values for test data
        "activity": [
            89.0,
            86.0,
        ],  # Target activity values for reference (not used for prediction)
    }
)


# Prepare the test dataset with Jaqpotpy
test_dataset = JaqpotpyDataset(
    df=X_test,
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

predictions = jaqpot_model.predict(test_dataset)
print(predictions)

# Conducts a randomization test to assess the model's robustness against randomization of target labels.
# This test involves shuffling the target labels in the training dataset multiple times (specified by n_iters).
# For each iteration, the model is retrained with the randomized targets, then evaluated on the original test set.
# This approach helps determine if the model's predictions are truly capturing relationships in the data
# (i.e., significantly better than a model trained on random labels) or if the performance is mainly due to chance.

jaqpot_model.randomization_test(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    n_iters=10,
)
