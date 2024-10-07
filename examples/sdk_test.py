from jaqpotpy.api.jaqpot_api_client import JaqpotApiClient
from jaqpotpy.jaqpot import Jaqpot
from jaqpotpy.descriptors.molecular import TopologicalFingerprint
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.api.openapi.models.dataset import Dataset
from jaqpotpy.api.openapi.models import DatasetType

import pandas as pd


jaqpot_api_client=JaqpotApiClient()

# Get a model by id
model = jaqpot_api_client.get_model_by_id(model_id=1842)  # 1812)
print(model)
# Get model summary
# model_summary = jaqpot.get_model_summary(model_id=1837)
# print(model_summary)


# Get shared models with organization
# shared_models = jaqpot.get_shared_models()
# print(shared_models)

# Get summary of shared models with organization
# shared_models_summary = jaqpot.get_shared_models_summary()
# print(shared_models_summary)

# Take a prediction with a model
# input_data = [{"SÎ¶MILES": "CC", "X1": 1, "X2": 2}]
# prediction = jaqpot.predict_with_model_sync(model_id=1812, dataset=input_data)
# print(prediction)

# Take prediction with a model and a csv file
# csv_path = "/Users/vassilis/Desktop/test_csv.csv"
# prediction = jaqpot.predict_with_csv(model_id=1812, csv_path=csv_path)
# print(prediction)


# Testing QsarToolBox
# get_model = jaqpot.get_model_by_id(model_id=6)
# print(get_model)

# Test QsarToolBox calculator
# prediction = jaqpot.qsartoolbox_calculator_predict_sync(
#     smiles="CC", calculatorGuid="c377150b-77ae-4f99-be14-357b85dd8d1f"
# )
# print(prediction)

# Test QsarToolBox Model
# prediction = jaqpot.qsartoolbox_qsar_model_predict_sync(
#     smiles="CC", qsarGuid="c377150b-77ae-4f99-be14-357b85dd8d1f"
# )
# print(prediction)

# Test QsarToolBox Profiler
# prediction = jaqpot.qsartoolbox_profiler_predict_sync(
#     smiles="CC", profilerGuid="723eb011-3e5b-4565-9358-4c3d8620ba5d"
# )
# print(prediction)
