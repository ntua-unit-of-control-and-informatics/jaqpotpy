from jaqpotpy.descriptors.molecular import TopologicalFingerprint
from jaqpotpy.models import MolecularSKLearn
from jaqpotpy.doa.doa import Leverage
from sklearn.linear_model import LinearRegression
from jaqpotpy import Jaqpot
import pandas as pd 
import numpy as np

# Create a DataFrame with random values
external_df = pd.DataFrame(np.random.randint(0, 10, size=(5, 3)), columns=['A', 'B', 'C'])
y = pd.DataFrame([6.5, 3.4, 2.4, 1.5, 7], columns=['Y']) 

lr = LinearRegression()
lr.fit(external_df, y)

validation_x = pd.DataFrame({'A': [1,10,100], 'B': [2,20,200], 'C': [3,30,300]})
predictions = lr.predict(validation_x)  # Get predictions for validation_x

# # Upload to local jaqpot
file_path = '/Users/vassilis/Desktop/jaqpot_api_key.txt'
with open(file_path, 'r') as file:
    api_key = file.read()

jaqpot = Jaqpot("http://localhost:8080/jaqpot/services/")
jaqpot.set_api_key(api_key)
jaqpot.deploy_sklearn(model=lr, X=external_df, y=y, 
                      title='Model only with external features', 
                      description='Model only with external features')