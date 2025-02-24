from typing import List
from jaqpot_api_client import DockerConfig, Feature, ModelVisibility, ModelType

from jaqpotpy import Jaqpot
from jaqpotpy.models import Model


class DockerModel(Model):
    def __init__(
        self,
        independent_features: List[Feature],
        dependent_features: List[Feature],
        docker_config: DockerConfig,
    ):
        self.independent_features = independent_features
        self.dependent_features = dependent_features
        self.docker_config = docker_config
        self.model_type = ModelType.DOCKER

    def deploy_on_jaqpot(
        self, jaqpot: Jaqpot, name: str, description: str, visibility: ModelVisibility
    ):
        """
        Deploy the model on Jaqpot.

        Args:
            jaqpot: The Jaqpot instance.
            name (str): The name of the model.
            description (str): The description of the model.
            visibility: The visibility of the model.
        """
        jaqpot.deploy_docker_model(
            model=self, name=name, description=description, visibility=visibility
        )
