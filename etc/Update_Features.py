import json
from dotenv import load_dotenv
from datetime import datetime
from jaqpot_python_sdk.jaqpot_api_client import JaqpotApiClient
from jaqpot_api_client import ModelApi, FeatureApi, PartiallyUpdateModelFeatureRequest
from os import environ

# --------------------------------------------------------------------
# This script is used to update the model features in the Jaqpot API
# --------------------------------------------------------------------
load_dotenv(".env")
# Get API keys from environment variables and print them

# Print available environment variables and model IDs for debugging
print("Environment variables loaded:")
if "JAQPOT_API_KEY" in environ:
    print(f"JAQPOT_API_KEY: {environ['JAQPOT_API_KEY'][:5]}...")
else:
    print("JAQPOT_API_KEY not found")


old_full_model_id = 2116
# old_lite_model_id = 2115

new_full_model_id = 2124
# new_lite_model_id = 2123


# old_model_id = old_full_model_id
# new_model_id = new_full_model_id

api_client = JaqpotApiClient().http_client
feature_api = FeatureApi(api_client=api_client)


# Function to update features from previous model to new model
def update_features(previous_model, new_model, feature_api, new_model_id, feature_type):
    features = (
        new_model.independent_features
        if feature_type == "independent"
        else new_model.dependent_features
    )
    previous_features = (
        previous_model.independent_features
        if feature_type == "independent"
        else previous_model.dependent_features
    )

    for feature in features:
        feature_name = feature.name
        try:
            previous_feature = next(
                f for f in previous_features if f.name == feature_name
            )
            # Update the new feature with the previous feature data
            update_request = PartiallyUpdateModelFeatureRequest(
                name=feature_name,
                feature_type=previous_feature.feature_type,
                description=previous_feature.description,
                units=previous_feature.units,
            )
            feature_api.partially_update_model_feature(
                new_model_id, feature.id, update_request
            )
            print(f"Updated {feature_type} feature: {feature_name}")
        except StopIteration:
            print(
                f"Warning: {feature_type} feature '{feature_name}' not found in previous model"
            )


# Update features for full model
print(f"\nUpdating features for Full Model (ID: {new_full_model_id})")
previous_model = ModelApi(api_client=api_client).get_model_by_id(old_full_model_id)
new_model = ModelApi(api_client=api_client).get_model_by_id(new_full_model_id)
update_features(
    previous_model, new_model, feature_api, new_full_model_id, "independent"
)
update_features(previous_model, new_model, feature_api, new_full_model_id, "dependent")

# Update features for lite model
# print(f"\nUpdating features for Lite Model (ID: {new_lite_model_id})")
# previous_model = ModelApi(api_client=api_client).get_model_by_id(old_lite_model_id)
# new_model = ModelApi(api_client=api_client).get_model_by_id(new_lite_model_id)
# update_features(
#     previous_model, new_model, feature_api, new_lite_model_id, "independent"
# )
# update_features(previous_model, new_model, feature_api, new_lite_model_id, "dependent")
