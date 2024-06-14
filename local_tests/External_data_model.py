from jaqpotpy.descriptors.molecular import TopologicalFingerprint
from jaqpotpy.datasets import MolecularTabularDataset
from jaqpotpy.models import MolecularSKLearn
from jaqpotpy.doa.doa import Leverage
from sklearn.linear_model import LinearRegression
from jaqpotpy import Jaqpot
import pandas as pd 
import numpy as np

# Create a DataFrame with random values
external_df = pd.DataFrame(np.random.randint(0, 10, size=(5, 3)), columns=['A', 'B', 'C'])

smiles = ['CCCCCCC', 'C1CCC1', 'CC(=O)CC', 'CCCCCCCC', 'CCCC1CC1']
smiles_df = pd.DataFrame(smiles, columns=['SMILES'])
y = pd.DataFrame([6.5, 3.4, 2.4, 1.5, 7], columns=['Y']) 

final_df = pd.concat([smiles_df, external_df, y], axis=1)
final_df.to_csv('/Users/vassilis/Documents/GitHub/jaqpotpy/external_df.csv', index=False)

featurizer = TopologicalFingerprint()
training_dataset = MolecularTabularDataset(path='/Users/vassilis/Documents/GitHub/jaqpotpy/external_df.csv',
                                           x_cols=['A','B','C'],
                                           y_cols=['Y'],
                                           smiles_col="SMILES",
                                           featurizer=featurizer)
training_dataset.create()

lr = LinearRegression()
model = MolecularSKLearn(dataset=training_dataset, doa = None, model=lr)
fitted_model = model.fit()


smiles_test = ['CC', 'CCCCCCCCCCC']
external_data = {'B': (2.5, 5), 'C': (3.5, 7), 'A': (1.5, 3)}
fitted_model(smiles_test, external_data)
fitted_model.prediction

# # Upload to local jaqpot
file_path = '/Users/vassilis/Desktop/jaqpot_api_key.txt'
with open(file_path, 'r') as file:
    api_key = file.read()

jaqpot = Jaqpot("http://localhost:8080/jaqpot/services/")
jaqpot.set_api_key(api_key)
fitted_model.deploy_on_jaqpot(jaqpot=jaqpot,
                        description="Test smiles and external model",
                        model_title="Test smiles and external model")