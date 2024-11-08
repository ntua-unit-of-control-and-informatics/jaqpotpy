import pandas as pd
from sklearn.datasets import make_classification
from jaqpotpy.datasets import JaqpotpyDataset
from sklearn.linear_model import LogisticRegression
from jaqpotpy.models import SklearnModel

# Generate a small binary classification dataset
X, y = make_classification(n_samples=100, n_features=4, random_state=42)

# Create a DataFrame with the features and target
# Store the features in columns "X1", "X2", "X3", "X4" and the target in column "y".
df = pd.DataFrame(X, columns=["X1", "X2", "X3", "X4"])
df["y"] = y

# Initialize a JaqpotpyDataset with the DataFrame
# Specify the feature columns and the target column, and define the task as binary classification.
dataset = JaqpotpyDataset(
    df=df,
    x_cols=["X1", "X2", "X3", "X4"],
    y_cols=["y"],
    task="binary_classification",
)

# Wrap the scikit-learn model with Jaqpotpy's SklearnModel
# Use Logistic Regression as the classification model.
jaqpot_model = SklearnModel(dataset=dataset, model=LogisticRegression())

# Fit the model to the dataset
# Train the Logistic Regression model using the provided dataset.
jaqpot_model.fit()

# Upload mdodel on Jaqpot
# First import Jaqpot class from jaqpotpy
from jaqpotpy import Jaqpot  # noqa: E402

# Next, create an instance of Jaqpot
jaqpot = Jaqpot()
# Then login to Jaqpot. jaqpot.login will prompt you to enter
# an authorization code that you will receive from your
#  browser after you login to Jaqpot.
jaqpot.login()
# Deploy the model on Jaqpot
# Give a title to your model and a description. Also define if
# if the model will be public or private. If it is private, only you
# will be able to see it and use it to take predictions.
jaqpot_model.deploy_on_jaqpot(
    jaqpot=jaqpot,
    name="My first Jaqpot Model",
    description="This is my first attempt to train and upload a Jaqpot model.",
    visibility="PRIVATE",
)
