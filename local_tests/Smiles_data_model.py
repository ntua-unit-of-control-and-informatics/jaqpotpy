import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from jaqpotpy.descriptors.molecular import TopologicalFingerprint
from jaqpotpy.datasets import SmilesDataset
from jaqpotpy.models import MolecularSKLearn
from jaqpotpy.doa.doa import MeanVar, Leverage
from jaqpotpy import Jaqpot

# Create a DataFrame with random values
external_df = pd.DataFrame(np.random.randint(1, 100, size=(5, 3)), columns=['A', 'B', 'C'])

smiles = ['CCCCCCC', 'C1CCC1', 'CC(=O)CC', 'CCCCCCCC', 'CCCC1CC1']

y = [6.5, 3.4, 2.4, 1.5, 7]

featurizer = TopologicalFingerprint()
training_dataset = SmilesDataset(smiles=smiles, y=y, featurizer=featurizer)
training_dataset.create()
type(training_dataset.__get_Y__())
lr = LinearRegression()
model = MolecularSKLearn(dataset=training_dataset, doa=MeanVar(), model=lr)
fitted_model = model.fit()

smiles_test = ['CC']
fitted_model(smiles_test)

# Upload to local jaqpot
file_path = '/Users/vassilis/Desktop/jaqpot_api_key.txt'
with open(file_path, 'r') as file:
    api_key = file.read()

jaqpot = Jaqpot("http://localhost:8080/jaqpot/services/")
jaqpot.set_api_key(api_key)
fitted_model.deploy_on_jaqpot(jaqpot=jaqpot,
                        description="Smiles Only Model",
                        model_title="Smiles Only Model")