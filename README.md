[![Build and test](https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy/actions/workflows/build.yml/badge.svg)](https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy/actions/workflows/build.yml) [![Publish to PyPI 📦](https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy/actions/workflows/pipy_release.yml/badge.svg)](https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy/actions/workflows/pipy_release.yml)

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

### Releasing to PyPI
Releasing the latest version of jaqpotpy to PyPI is automated via GitHub Actions. When you create a new release on GitHub, the workflow is triggered to publish the latest version to the PyPI registry.

#### How to Release
1. Follow Semantic Versioning: Use the format 1.XX.YY where XX is the minor version and YY is the patch version.

2. Create a New Release:

- Navigate to the repository’s Releases section.
- Click on Draft a new release.

3. Generate Release Notes:

Use GitHub’s feature to automatically generate release notes or customize them as needed.

4. Publish the Release:

Once published, the GitHub Action will automatically upload the latest files to the PyPI registry.

After the release is completed, the new version of jaqpotpy will be available on PyPI and ready for users to install.
