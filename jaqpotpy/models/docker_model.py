from typing import List
from jaqpot_api_client import DockerConfig, Feature, ModelVisibility, ModelType

from jaqpotpy.models import Model


class DockerModel(Model):
    """
    A representation of a Docker-based model in Jaqpot.

    This class extends the `Model` class and is designed for deploying models encapsulated within
    Docker containers. It requires independent and dependent features, along with a Docker configuration.

    Attributes:
        independent_features (List[Feature]): The features used as inputs for the model.
        dependent_features (List[Feature]): The target/output features of the model.
        docker_config (DockerConfig): The Docker container configuration for the model.
        model_type (ModelType): The type of the model, set to `ModelType.DOCKER`.
    """

    def __init__(
        self,
        independent_features: List[Feature],
        dependent_features: List[Feature],
        docker_config: DockerConfig,
    ):
        """
        Initializes a DockerModel instance.

        Args:
            independent_features (List[Feature]): The features used as inputs for the model.
            dependent_features (List[Feature]): The target/output features of the model.
            docker_config (DockerConfig): The Docker container configuration.
        """
        super().__init__()
        self.independent_features = independent_features
        self.dependent_features = dependent_features
        self.docker_config = docker_config
        self.model_type = ModelType.DOCKER  # Ensuring the model type is explicitly set

    def deploy_on_jaqpot(
        self, jaqpot, name: str, description: str, visibility: ModelVisibility
    ) -> None:
        """
        Deploys the Docker-based model to Jaqpot.

        This method registers the model in the Jaqpot platform and makes it available based on the provided visibility.

        Args:
            jaqpot (Jaqpot): The Jaqpot instance responsible for model deployment.
            name (str): The name to be assigned to the model.
            description (str): A brief description of the model.
            visibility (ModelVisibility): The access level of the model (e.g., public, private, organizational).

        Raises:
            ValueError: If any required parameter is invalid.
        """
        if not name or not isinstance(name, str):
            raise ValueError("Model name must be a non-empty string.")
        if not description or not isinstance(description, str):
            raise ValueError("Model description must be a non-empty string.")
        if not isinstance(visibility, ModelVisibility):
            raise ValueError(
                "Invalid visibility type. Expected a ModelVisibility instance."
            )

        jaqpot.deploy_docker_model(
            model=self, name=name, description=description, visibility=visibility
        )
