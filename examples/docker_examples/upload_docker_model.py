from jaqpot_api_client import Feature, DockerConfig, ModelVisibility, FeatureType
from jaqpotpy.models.docker_model import DockerModel
from jaqpotpy import Jaqpot

# Define independent and dependent features
independent_features = [
    Feature(key="x1", name="X1", feature_type=FeatureType.FLOAT),
    Feature(key="x2", name="X2", feature_type=FeatureType.FLOAT),
]

dependent_features = [Feature(key="y", name="Y", feature_type=FeatureType.FLOAT)]

# Create a dummy DockerConfig (update values as needed)
docker_config = DockerConfig(app_name="fake-model", docker_image="fake-model-image")

# Instantiate a DockerModel
jaqpot_model = DockerModel(
    independent_features=independent_features,
    dependent_features=dependent_features,
    docker_config=docker_config,
)

# Create an instance of Jaqpot (ensure local Jaqpot is running)
jaqpot = Jaqpot(
    base_url="http://localhost.jaqpot.org",
    app_url="http://localhost.jaqpot.org:3000",
    login_url="http://localhost.jaqpot.org:8070",
    api_url="http://localhost.jaqpot.org:8080",
    keycloak_realm="jaqpot-local",
    keycloak_client_id="jaqpot-local-test",
)

# Login to Jaqpot (requires authorization from browser)
jaqpot.login()

# Deploy the model on Jaqpot
jaqpot_model.deploy_on_jaqpot(
    jaqpot=jaqpot,
    name="Fake model",
    description="This is my first attempt to train and upload a Jaqpot model.",
    visibility=ModelVisibility.PRIVATE,  # Fixed to use the correct enum
)
