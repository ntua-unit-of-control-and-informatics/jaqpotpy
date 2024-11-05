from dotenv import load_dotenv

load_dotenv(".env")
from jaqpotpy.api.jaqpot_api_client import JaqpotApiClient  # noqa: E402

jaqpot = JaqpotApiClient()

# Get a model by id
model = jaqpot.get_model_by_id(model_id=1691)  # 1812)
print(model)
# 
# # Get model summary
# model_summary = jaqpot.get_model_summary(model_id=1691)
# print(model_summary)
# 
# # Get shared models with organization
# shared_models = jaqpot.get_shared_models()
# print(shared_models)
# 
# # # Take a synchronous prediction with a model
# input_data = [{"SMILES": "CC", "X1": 1, "X2": 2}]
# prediction = jaqpot.predict_sync(model_id=1852, dataset=input_data)
# print(prediction)
# 
# # Take an asynchronous prediction with a model
# input_data = [{"SMILES": "CC", "X1": 1, "X2": 2}]
# prediction = jaqpot.predict_async(model_id=1812, dataset=input_data)
# 
# # Take prediction with a model and a csv file
# csv_path = "/Users/vassilis/Desktop/test_csv.csv"
# prediction = jaqpot.predict_with_csv_sync(model_id=1812, csv_path=csv_path)
# print(prediction)
# 
# 
# # Testing QsarToolBox
# get_model = jaqpot.get_model_by_id(model_id=6)
# print(get_model)
# 
# # Test QsarToolBox calculator
# prediction = jaqpot.qsartoolbox_calculator_predict_sync(
#     smiles="CC", calculator_guid="1804a854-9041-4495-9931-7414c22a5e49"
# )
# print(prediction)
# 
# # Test QsarToolBox Model
# prediction = jaqpot.qsartoolbox_qsar_model_predict_sync(
#     smiles="CC", qsar_guid="c377150b-77ae-4f99-be14-357b85dd8d1f"
# )
# print(prediction)
# 
# # Test QsarToolBox Profiler
# prediction = jaqpot.qsartoolbox_profiler_predict_sync(
#     smiles="CC", profiler_guid="723eb011-3e5b-4565-9358-4c3d8620ba5d"
# )
# print(prediction)
