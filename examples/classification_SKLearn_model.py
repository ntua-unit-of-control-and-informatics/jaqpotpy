import pandas as pd
from sklearn.datasets import make_classification
from jaqpotpy.datasets import JaqpotpyDataset
from sklearn.linear_model import LogisticRegression
from jaqpotpy.models.sklearn import SklearnModel

# Step 1: Generate a small binary classification dataset
# This creates a dataset with 100 samples, each having 4 features.
X, y = make_classification(n_samples=100, n_features=4, random_state=42)

# Step 2: Create a DataFrame with the features and target
# We store the features in columns "X1", "X2", "X3", "X4" and the target in column "y".
df = pd.DataFrame(X, columns=["X1", "X2", "X3", "X4"])
df["y"] = y

# Step 3: Initialize a JaqpotpyDataset with the DataFrame
# Specify the feature columns and the target column, and define the task as binary classification.
dataset = JaqpotpyDataset(
    df=df,
    x_cols=["X1", "X2", "X3", "X4"],
    y_cols=["y"],
    task="binary_classification",
)

# Step 4: Wrap the scikit-learn model with Jaqpotpy's SklearnModel
# Here, we use Logistic Regression as the classification model.
jaqpot_model = SklearnModel(dataset=dataset, model=LogisticRegression())

# Step 5: Fit the model to the dataset
# This trains the Logistic Regression model using the provided dataset.
jaqpot_model.fit()
