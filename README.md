[![Build and test](https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy/actions/workflows/build.yml/badge.svg)](https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy/actions/workflows/build.yml) [![Publish to PyPI 📦](https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy/actions/workflows/pipy_release.yml/badge.svg)](https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy/actions/workflows/pipy_release.yml)

# Jaqpotpy

The jaqpotpy library enables you to upload and deploy machine learning models to the Jaqpot platform. Once uploaded, you can manage, document, and share your models via the Jaqpot user interface at **https://app.jaqpot.org**. You can also make predictions online or programmatically using the Jaqpot API.

## Getting Started

### Prerequisites

- Python 3.10
- An account on **https://app.jaqpot.org**

### Installation

Install jaqpotpy using pip:

```bash
pip install jaqpotpy
```

### Logging In

To use jaqpotpy, you need to log in to the Jaqpot platform. You can log in using the login() method

#### Login with Username and Password

```python
from jaqpotpy import Jaqpot

jaqpot = Jaqpot()
jaqpot.login() # follow the steps here to login through the command line 
```

### Model Training and Deployment

Follow these steps to train and deploy your model on Jaqpot:

	1. Train your model using pandas DataFrame as input.
	2. Deploy the trained model using the deploy_on_jaqpot function.

#### Example Code

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.models import SklearnModel
from jaqpotpy import Jaqpot

np.random.seed(42)
X1 = np.random.rand(100)
X2 = np.random.rand(100)
ACTIVITY = 2 * X1 + 3 * X2 + np.random.randn(100) * 0.1
df = pd.DataFrame({"X1": X1, "X2": X2, "ACTIVITY": ACTIVITY})
y_cols = ["ACTIVITY"]
x_cols = ["X1", "X2"]

# Step 1: Create a Jaqpotpy dataset
dataset = JaqpotpyDataset(df=df, y_cols=y_cols, x_cols=x_cols, task="regression")

# Step 2: Build a model
rf = RandomForestRegressor(random_state=42)
myModel = SklearnModel(dataset=dataset, model=rf)
myModel.fit()

# Step 3: Upload the model on Jaqpot
jaqpot = Jaqpot()
jaqpot.login()
myModel.deploy_on_jaqpot(
    jaqpot=jaqpot,
    name="Demo: Regression",
    description="This is a description",
    visibility="PRIVATE",
)

```

The function will provide you with the model ID that you can use to manage your model through the user interface and API.

Result:
```text
<DATE> - INFO - Model has been successfully uploaded. The url of the model is https://app.jaqpot.org/dashboard/models/<ModelID>
```

#### Managing Your Models

You can further manage your models through the Jaqpot user interface at https://app.jaqpot.org. This platform allows you to view detailed documentation, share models with your contacts, and make predictions.
