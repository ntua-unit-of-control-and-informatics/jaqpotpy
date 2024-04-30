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

model = MolecularSKLearn(training_dataset, doa=Leverage(), model=lr, eval=None)
fitted = model.fit()

# Infer
input_dataset = ['CC']
fitted(input_dataset)
print(fitted.prediction)

#Upload to local jaqpot
api_key = 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJ3Ujh3X1lGOWpKWFRWQ2x2VHF1RkswZkctQXROQUJsb3FBd0N4MmlTTWQ4In0.eyJleHAiOjE3MTQ2NDE5NTMsImlhdCI6MTcxNDQ2OTE1MywiYXV0aF90aW1lIjoxNzE0NDY5MTUyLCJqdGkiOiIwNDU3YTBiYi02OWMxLTRkOTktYjAzZS02NjRlNDMwMGYwZGMiLCJpc3MiOiJodHRwczovL2xvZ2luLmphcXBvdC5vcmcvYXV0aC9yZWFsbXMvamFxcG90IiwiYXVkIjpbImJyb2tlciIsImFjY291bnQiXSwic3ViIjoiYzMzZGVhMTgtYzA3Zi00YmNhLTgzZjctOTliYmNlMmE5YWNiIiwidHlwIjoiQmVhcmVyIiwiYXpwIjoiamFxcG90LXVpLWNvZGUiLCJub25jZSI6IjUzZDgzNWNlMWM4ZmZiZWZhY2EzMzI3ODM3Mzg3YWUxYzh2cmtxTWF2Iiwic2Vzc2lvbl9zdGF0ZSI6ImJkOTVjOTQxLWVkNzYtNDJlZS1hOGY0LTgwODM1YzAyNmVmZCIsImFsbG93ZWQtb3JpZ2lucyI6WyInKiciLCIqIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYnJva2VyIjp7InJvbGVzIjpbInJlYWQtdG9rZW4iXX0sImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoib3BlbmlkIGphcXBvdC1hY2NvdW50cyBlbWFpbCBwcm9maWxlIHdyaXRlIHJlYWQiLCJzaWQiOiJiZDk1Yzk0MS1lZDc2LTQyZWUtYThmNC04MDgzNWMwMjZlZmQiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwibmFtZSI6IlBlcmlrbGlzIFRzaXJvcyIsInByZWZlcnJlZF91c2VybmFtZSI6InBlcmlrbGlzdHMiLCJnaXZlbl9uYW1lIjoiUGVyaWtsaXMiLCJsb2NhbGUiOiJlbiIsImZhbWlseV9uYW1lIjoiVHNpcm9zIiwiZW1haWwiOiJwdHNpcm9zdHNpYkBnbWFpbC5jb20ifQ.cxVk_2OHmnPvaidiYPgt7GkCRVwL4Enhy6P7SeQ6LanuWpO-W4vkd_p7RD8CgqpdC0sHK-wQRfMwYdan1bfa4uQRpT8wpAxrvnNqkRmYeG0HUwe7O7QSiWaWmt5YtwsCGmVRZGSsA8TqXms3p-vVsTqysQ-M2R8TjRD4jHzNePYGu47ud_CS_2KGY3I4w7dRK-_SOO7CiHLpiF-fjq6FO6RrAOsyB-XlIxYXu_8_W5ZU1uMCUpRTE5odcAP53gk2nF0m31v2Xj31ZhSlJpA_PQxhqO2o-2J72zk6lDH-BiOIYLSV3YOaXrwyLUeUe67p3443M-slJ1W4N2WIee48Vw'  # set your api key here

jaqpot = Jaqpot("http://localhost:8080/jaqpot/services/")
jaqpot.set_api_key(api_key)
fitted.deploy_on_jaqpot(jaqpot=jaqpot,
                        description="peri",
                        model_title="test_model")
