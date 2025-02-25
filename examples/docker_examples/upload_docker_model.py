from jaqpot_api_client import Feature, DockerConfig, ModelVisibility, FeatureType
from jaqpotpy.models.docker_model import DockerModel
from jaqpotpy import Jaqpot

# Define independent and dependent features
independent_features = [
    Feature(
        key="numGenerations", name="numGenerations", feature_type=FeatureType.FLOAT
    ).to_dict(),
]

dependent_features = [
    Feature(
        key="prediction", name="Prediction", feature_type=FeatureType.FLOAT
    ).to_dict(),
    Feature(key="smiles", name="SMILES", feature_type=FeatureType.STRING).to_dict(),
]

# Create a dummy DockerConfig (update values as needed)
docker_config = DockerConfig(
    app_name="gflownet", docker_image="upcintua/jaqpot-gflownet"
)

# Instantiate a DockerModel
jaqpot_model = DockerModel(
    independent_features=independent_features,
    dependent_features=dependent_features,
    docker_config=docker_config,
)

# Create an instance of Jaqpot (ensure local Jaqpot is running)
jaqpot = Jaqpot()

# Login to Jaqpot (requires authorization from browser)
jaqpot.login()

# Deploy the model on Jaqpot
jaqpot_model.deploy_on_jaqpot(
    jaqpot=jaqpot,
    name="Gflownet model",
    description="This is my first attempt to train and upload a Jaqpot model.",
    visibility=ModelVisibility.PRIVATE,
)
