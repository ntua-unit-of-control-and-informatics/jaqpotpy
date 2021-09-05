# Jaqpot 

Jaqpot platform enables sklearn, xgBoost and other models developed in python to be accessible through a user interface, that allows extensive documentation of the models and sharing through your contacts.


## jaqpotpy

jaqpotpy enables model deployment with a simple command. 

### First register to Jaqpot through **https://app.jaqpot.org**

`jaqpot = Jaqpot()`  initializes jaqpot upon the standard available API that 
is integrated with the application and user interface at **https://app.jaqpot.org/** .



 ### Let jaqpot know who you are

Login and have access on the jaqpot services

In order to do so you can use the functions:

* `jaqpot.login('username', 'password')`

Will login and set the api key that is needed.

* `jaqpot.request_key('username', 'password')`

Same as above you request the key and set it on jaqpot

* `jaqpot.request_key_safe()`

Will ask the user for the username and password by hidding the password if 
jaqpot is used through a jupiter notebook etc

### Set Key without login

Some users may have logged in through google or github. At the account page 
a user can find an api key that can be used in order to have access on the services.
These keys have short life and should be updated on each login.

* `jaqpot.set_api_key("api_key")`

#### Get the key from user interface


### Model training and deployment


An example code that demonstrates a model deployemnt.


:::caution

Warning! One of the things that may differ from simpler training and validation routes is that you need to train your model with a pandas dataframe as input and not with Numpy arrays!

:::

```python
from jaqpotpy import Jaqpot
import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv('/path/to/gdp.csv')
lm = LinearRegression()

y = df['GDP']
X = df[['LFG', 'EQP', 'NEQ', 'GAP']]

model = lm.fit(X=X, y=y)

jaqpot.deploy_sklearn(model, X, y, title="Title", description="Describe")
```

The function will inform you about the model id that is created and is available through the user interface and the API.


:::info Result


- INFO - Model with id: <model_id> created. Visit the application to proceed

:::


### Continue furter through the jaqpot user interface




