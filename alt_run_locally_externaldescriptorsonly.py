# This script creates a model with new classes that are designed when only numerical descriptors are used

from jaqpotpy.datasets.dataset_base import NumericalVectorDataset
from jaqpotpy.doa.doa import Leverage
from jaqpotpy.models import NumericalSKLearn
from sklearn.linear_model import LinearRegression
from jaqpotpy import Jaqpot

from typing import List, Optional

# Train model

# Input through a csv
#    ****************************************************
path_to_csv = r'C:\Users\harri\Downloads\ExternalOnlytar.csv'
training_dataset = NumericalVectorDataset(path=path_to_csv)
#    ****************************************************


# Input through tensors
#    ****************************************************
vectors = [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]
targets = [0.1, 0.2, 0.3]
training_dataset = NumericalVectorDataset(vectors=vectors, targets=targets)
#    ****************************************************

training_dataset.create()
lr = LinearRegression()

print(training_dataset.df)

model = NumericalSKLearn(training_dataset, doa=None, model=lr, eval=None)
fitted = model.fit()


 #Upload to local jaqpot

jaqpot = Jaqpot()

jaqpot.request_key('guest','guest')
 
fitted.deploy_on_jaqpot(jaqpot=jaqpot, X=fitted.X, y=fitted.Y, description="ALEX", model_title="RANDOM")  #GIVES ERROR 

# Infer

external_features = {'feature1': [1.5, 2.3, 3.1], 'feature2': [0.6, 0.8, 0.9]}
fitted(external_features)
print(fitted.prediction)
print('this is a test')



