import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

from jaqpotpy import Jaqpot
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.models import SklearnModel

data = {
    "X1": [0.5, 1, 1.5, 3, 2, 1],
    "X2": [1.5, 1, 0.5, 0.5, 2, 2.5],
    "RESULT": [0, 0, 1, 1, 1, 1],
}

df = pd.DataFrame(data)

y_cols = ["RESULT"]
x_cols = ["X1", "X2"]
dataset = JaqpotpyDataset(
    df=df, y_cols=y_cols, x_cols=x_cols, task="BINARY_CLASSIFICATION"
)

lr_model = LogisticRegression()

model = SklearnModel(dataset=dataset, doa=None, model=lr_model)
model.fit()

# Upload locally
jaqpot = Jaqpot(
    base_url="http://localhost.jaqpot.org",
    app_url="http://localhost.jaqpot.org:3000",
    login_url="http://localhost.jaqpot.org:8070",
    api_url="http://localhost.jaqpot.org:8080",
    keycloak_realm="jaqpot-local",
    keycloak_client_id="jaqpot-local-test",
)

# jaqpot = Jaqpot()
jaqpot.login()
model.deploy_on_jaqpot(
    jaqpot=jaqpot,
    name="Alex logistic regression",
    description="Test logistic regression",
    visibility="PRIVATE",
)
