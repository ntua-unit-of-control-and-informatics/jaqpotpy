[![Build and test](https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy/actions/workflows/ci.yml/badge.svg)](https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy/actions/workflows/ci.yml) [![Publish to PyPI 📦](https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy/actions/workflows/pipy_release.yml/badge.svg)](https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy/actions/workflows/pipy_release.yml)

# Jaqpotpy

The jaqpotpy library enables you to upload and deploy machine learning models to the Jaqpot platform. Once uploaded, you can manage, document, and share your models via the Jaqpot user interface at **https://app.jaqpot.org**. You can also make predictions online or programmatically using the Jaqpot API.

## Getting Started

### Prerequisites

- Python 3.x
- An account on **https://app.jaqpot.org**

### Installation

Install jaqpotpy using pip:

```bash
pip install jaqpotpy
```

### Logging In

To use jaqpotpy, you need to log in to the Jaqpot platform. You can log in using your username and password or by setting an API key.

#### Login with Username and Password

```python
from jaqpotpy import Jaqpot

jaqpot = Jaqpot()
jaqpot.login('your_username', 'your_password') 
```

#### Request and Set API Key

You can request an API key and set it:
```python
jaqpot.request_key('your_username', 'your_password')
```
or
```python
jaqpot.request_key_safe()  # Prompts for username and password securely
```
#### Set API Key Directly

If you already have an API key (you can retrieve one from https://app.jaqpot.org), you can set it directly:

```python
jaqpot.set_api_key("your_api_key")
```

### Model Training and Deployment

Follow these steps to train and deploy your model on Jaqpot:

	1. Train your model using pandas DataFrame as input.
	2. Deploy the trained model using the deploy_on_jaqpot function.

#### Example Code
_Note: Ensure you use a pandas DataFrame for training your model._

```python
from jaqpotpy import Jaqpot
import pandas as pd
from sklearn.linear_model import LinearRegression

# Initialize Jaqpot
jaqpot = Jaqpot()

# Load your data
df = pd.read_csv('/path/to/gdp.csv')

# Train your model
lm = LinearRegression()
y = df['GDP']
X = df[['LFG', 'EQP', 'NEQ', 'GAP']]
model = lm.fit(X=X, y=y)

# Deploy the model on Jaqpot
jaqpot.deploy_sklearn(model, X, y, title="GDP Model", description="Predicting GDP based on various factors")
```

The function will provide you with the model ID that you can use to manage your model through the user interface and API.

Result:
```text
- INFO - Model with ID: <model_id> created. Visit the application to proceed.
```

#### Managing Your Models

You can further manage your models through the Jaqpot user interface at https://app.jaqpot.org. This platform allows you to view detailed documentation, share models with your contacts, and make predictions.
