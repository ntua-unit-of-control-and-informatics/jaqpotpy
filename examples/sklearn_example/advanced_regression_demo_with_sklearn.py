# Import necessary libraries for data handling, model training, and deployment
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from jaqpotpy.datasets import JaqpotTabularDataset
from jaqpotpy.models import SklearnModel
from jaqpotpy import Jaqpot
from jaqpotpy.doa import (
    MeanVar,
    BoundingBox,
    Leverage,
)  # Domain of Applicability (DOA) methods

# Define the path to the dataset
path = "examples/example_datasets/cytotoxicity_data.csv"  # this needs update

# Load dataset and preprocess column names
df = pd.read_csv(path)
df = df.rename(
    columns={"model_2_values": "cell_viability"}
)  # Rename target column for clarity
df = df.drop(columns=["type", "bmd_id"])  # Drop irrelevant columns
X, y = (
    df.drop("cell_viability", axis=1),
    df["cell_viability"],
)  # Split into features (X) and target (y)
X["func"] = X["func"].fillna("no_func")  # Fill missing values in 'func' column

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Combine X and y training/test sets back into dataframes for compatibility with Jaqpotpy
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Specify the feature (x_cols) and target (y_cols) columns for modeling
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

# Identify categorical columns for encoding
cat_cols = [
    "Substance",
    "size_class",
    "cell_type_general",
    "species",
    "assay",
    "media",
    "func",
]

# Initialize the JaqpotpyDataset for training
train_dataset = JaqpotTabularDataset(
    df=train_df,
    y_cols=y_cols,  # Specify target column
    x_cols=x_cols,  # Specify feature columns
    task="REGRESSION",  # Define task type
)

# Define preprocessing pipeline for feature columns
# - OneHotEncode categorical columns
# - Scale continuous columns with MinMaxScaler
column_transormer = ColumnTransformer(
    transformers=[
        ("OneHotEncoder", OneHotEncoder(), cat_cols),
        ("MinMaxScaler", MinMaxScaler(), ["layer", "time", "dose"]),
    ],
    remainder="passthrough",  # Leave non-specified columns unchanged
)

# Define the model: a Multi-Layer Perceptron (MLP) Regressor
model = MLPRegressor(
    solver="lbfgs",  # Solver for weight optimization
    random_state=42,  # Ensure reproducibility
    early_stopping=True,  # Stop training if validation score stops improving
    max_iter=5000,  # Maximum number of iterations
    hidden_layer_sizes=30,  # Single hidden layer with 30 neurons
)

# Define Domain of Applicability (DOA) checks for model evaluation
doa = [
    Leverage(),
    MeanVar(),
    BoundingBox(),
]  # Various DOA methods to ensure reliable predictions

# Wrap model in a Jaqpot SklearnModel object for easier integration with Jaqpotpy
jaqpotModel = SklearnModel(
    dataset=train_dataset,  # Training dataset
    doa=doa,  # DOA methods for prediction reliability
    model=model,  # MLP model defined above
    preprocess_x=column_transormer,  # Preprocessing pipeline
)
# Set random seed to have reproducibility of results
jaqpotModel.random_seed = 1231
jaqpotModel.fit()  # Train the model with the specified configuration

# Prepare the test dataset for model evaluation
test_dataset = JaqpotTabularDataset(
    df=test_df,
    y_cols=y_cols,  # Target column for test data
    x_cols=x_cols,  # Feature columns for test data
    task="REGRESSION",  # Specify regression task
)

# Evaluate model on the test dataset to assess performance
jaqpotModel.evaluate(test_dataset)


# Perform 10-fold cross-validation on the training dataset to validate model robustness
jaqpotModel.cross_validate(train_dataset, n_splits=10)

# Log in to Jaqpot platform for model deployment
jaqpot = Jaqpot()
jaqpot.login()  # Authenticate user (credentials required)

# Deploy trained model to Jaqpot platform for remote access and testing
jaqpotModel.deploy_on_jaqpot(
    jaqpot=jaqpot,  # Authenticated Jaqpot instance
    name="Test predictive model",  # Name of the deployed model
    description="Test",  # Short description for reference
    visibility="PRIVATE",  # Visibility setting on Jaqpot platform
)
