from jaqpotpy.descriptors.molecular import MACCSKeysFingerprint
from jaqpotpy.datasets import SmilesDataset
from jaqpotpy.doa.doa import Leverage
from jaqpotpy.models import MolecularSKLearn
from jaqpotpy import Jaqpot
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd


url = "https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv"
df = pd.read_csv(url, index_col=0)
print(df.head(5))

smiles = ['CCCCCCC', 'C1CCC1', 'CC(=O)CC', 'CCCCCCCC', 'CCCC1CC1']
y = [6.5, 3.4, 2.4, 1.5, 7. ]

featurizer = MACCSKeysFingerprint()
dataset = SmilesDataset(smiles=smiles, y=y, featurizer=featurizer, task='regression')
dataset.create()
lr = LinearRegression()

model = MolecularSKLearn(dataset, doa=Leverage(), model=lr, eval=None)
fitted = model.fit()

api_key = ''  # set your api key here

jaqpot = Jaqpot(baseUrl="http://localhost:8080/jaqpot/services/")
jaqpot.set_api_key(api_key)
fitted.deploy_on_jaqpot(jaqpot=jaqpot,
                        description="ALEX",
                        model_title="RANDOM")

model.prediction
