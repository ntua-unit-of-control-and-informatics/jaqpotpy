from jaqpotpy.descriptors.molecular import MACCSKeysFingerprint
from jaqpotpy.datasets import SmilesDataset
from jaqpotpy.datasets import MolecularTabularDataset
from jaqpotpy.datasets import dataset_base  #********************************
from jaqpotpy.datasets.dataset_base import NumericalVectorDataset   #********************************
from jaqpotpy.datasets.dataset_base import BaseDataset  #********************************
from jaqpotpy.datasets.dataset_base import MaterialDataset  #********************************
from jaqpotpy.doa.doa import Leverage
from jaqpotpy.models import MolecularSKLearn
from sklearn.linear_model import LinearRegression
from jaqpotpy import Jaqpot
import pandas as pd 


# Train model



sample_df = pd.read_csv(r"C:\Users\harri\Downloads\MolecularData.csv")

path_to_csv = r'C:\Users\harri\Downloads\MolecularData.csv'


featurizer = MACCSKeysFingerprint()

training_dataset = MolecularTabularDataset(path=path_to_csv, smiles_col='smiles', x_cols=['feature1', 'feature2'], y_cols=['activity'], featurizer=featurizer, task='regression')




#training_dataset = NumericalVectorDataset(x=x, y=y, task='regression') #********************************
training_dataset.create()
lr = LinearRegression()

print(training_dataset.df)

#model = MolecularSKLearn(training_dataset, doa=Leverage(), model=lr, eval=None)
model = MolecularSKLearn(training_dataset, doa=None, model=lr, eval=None)
fitted = model.fit()


 #Upload to local jaqpot

jaqpot = Jaqpot()

jaqpot.request_key('guest','guest')
 
fitted.deploy_on_jaqpot(jaqpot=jaqpot, description="ALEX", model_title="RANDOM")

# Infer

smiles = ['C(C(=O)O)N', 'CCO', 'CC(N)CO']
external_features = {'feature1': [1.5, 2.3, 3.1], 'feature2': [0.6, 0.8, 0.9]}
fitted(smiles,external_features)
print(fitted.prediction)
print('this is a test')



