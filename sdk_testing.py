from jaqpotpy.jaqpot import Jaqpot
from jaqpotpy.descriptors.molecular import TopologicalFingerprint
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.api.openapi.models.dataset import Dataset
from jaqpotpy.api.openapi.models import DatasetType

import pandas as pd


jaqpot = Jaqpot()
jaqpot.login()
# model = jaqpot.get_model_id(model_id=1812)
# shared_models = jaqpot.get_shared_models(organization_id=4)

input_data = [{"SMILES": ["CC"], "X1": [1], "X2": [2]}]
input_data = Dataset(
    type=DatasetType.PREDICTION,
    entry_type="ARRAY",
    input=input_data,
)

prediction = jaqpot.predict_with_model(model_id=1812, dataset=input_data)
print(prediction)
