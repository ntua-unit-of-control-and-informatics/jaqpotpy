from jaqpotpy.descriptors.molecular import MACCSKeysFingerprint
from jaqpotpy.datasets import SmilesDataset
from jaqpotpy.doa.doa import Leverage
from jaqpotpy.models import MolecularSKLearn
from sklearn.linear_model import LinearRegression
from jaqpotpy import Jaqpot

# Train model

smiles = ['CCCCCCC', 'C1CCC1', 'CC(=O)CC', 'CCCCCCCC', 'CCCC1CC1']
y = [6.5, 3.4, 2.4, 1.5, 7]

featurizer = MACCSKeysFingerprint()
training_dataset = SmilesDataset(smiles=smiles, y=y, featurizer=featurizer, task='regression')


training_dataset.create()
lr = LinearRegression()

print(training_dataset)

model = MolecularSKLearn(training_dataset, doa=Leverage(), model=lr, eval=None)
fitted = model.fit()

 #Upload to local jaqpot

jaqpot = Jaqpot()

jaqpot.request_key('guest','guest')
 
fitted.deploy_on_jaqpot(jaqpot=jaqpot, description="ALEX", model_title="RANDOM")

# Infer
input_dataset = ['CC']
fitted(input_dataset)
print(fitted.prediction)
print('this is a test')



