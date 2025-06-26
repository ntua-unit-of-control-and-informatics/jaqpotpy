import pandas as pd
from sklearn.datasets import make_regression
from jaqpotpy.datasets import JaqpotTabularDataset
from sklearn.linear_model import LinearRegression
from jaqpotpy.models import SklearnModel
from jaqpot_api_client.models.model_task import ModelTask

# Generate a small regression dataset
# This creates a dataset with 100 samples, each having 4 features and some noise.
X, y = make_regression(n_samples=100, n_features=4, noise=0.2, random_state=42)

# Create a DataFrame with the features and target
# We store the features in columns "X1", "X2", "X3", "X4" and the target in column "y".
df = pd.DataFrame(X, columns=["X1", "X2", "X3", "X4"])
df["y"] = y

# Initialize a JaqpotpyDataset with the DataFrame
# Specify the feature columns and the target column, and define the task as regression.
dataset = JaqpotTabularDataset(
    df=df,
    x_cols=["X1", "X2", "X3", "X4"],
    y_cols=["y"],
    task=ModelTask.REGRESSION,
)

# Wrap the scikit-learn model with Jaqpotpy's SklearnModel
# Here, we use Linear Regression as the regression model.
jaqpot_model = SklearnModel(dataset=dataset, model=LinearRegression())

# Fit the model to the dataset
# This trains the Linear Regression model using the provided dataset.
jaqpot_model.fit()

# Generate a small prediction dataset
# Create a new dataset with 5 samples, each having 4 features.
X_test, _ = make_regression(n_samples=5, n_features=4, noise=0.2, random_state=42)

# Create a DataFrame with the features
# Store the features in columns "X1", "X2", "X3", "X4".
df_test = pd.DataFrame(X_test, columns=["X1", "X2", "X3", "X4"])

# Initialize a JaqpotpyDataset for prediction
# Specify the feature columns and set y_cols to None since we are predicting.
test_dataset = JaqpotTabularDataset(
    df=df_test,
    x_cols=["X1", "X2", "X3", "X4"],
    y_cols=None,
    task=ModelTask.REGRESSION,
)

# Use the trained model to make predictions on the new dataset
# Predict the target values for the new dataset using the trained jaqpot_model model.
predictions = jaqpot_model.predict(test_dataset)

# Print the predictions
print(predictions)
