import webbrowser

from jaqpot_api_client.api import LargeModelApi
from keycloak import KeycloakOpenID

import jaqpotpy
from jaqpotpy.api.get_installed_libraries import get_installed_libraries
from jaqpot_python_sdk.jaqpot_api_client_builder import JaqpotApiHttpClientBuilder
from jaqpotpy.api.model_to_b64encoding import model_to_b64encoding
from jaqpot_api_client.api.model_api import ModelApi
from jaqpot_api_client.models.feature import Feature
from jaqpot_api_client.models.feature_type import FeatureType
from jaqpot_api_client.models.model import Model
from jaqpot_api_client.models.model_task import ModelTask
from jaqpot_api_client.models.model_type import ModelType
from jaqpot_api_client.models.model_visibility import ModelVisibility

from jaqpotpy.aws.s3 import upload_file_to_s3_presigned_url
from jaqpotpy.helpers.logging import init_logger
from jaqpotpy.helpers.url_utils import add_subdomain
from jaqpotpy.models.docker_model import DockerModel
from jaqpotpy.models.torch_models.torch_onnx import TorchONNXModel

ENCODING = "utf-8"
MAX_INLINE_MODEL_SIZE = 5 * 1024 * 1024  # 5MB


class Jaqpot:
    """Deploys sklearn and PyTorch models on Jaqpot.

    This class provides methods to log in to Jaqpot using Keycloak and deploy
    machine learning models (sklearn and PyTorch) on the Jaqpot platform.

    Parameters
    ----------
    base_url : str, optional
        The base URL on which Jaqpot services are deployed. Default is "https://jaqpot.org".
    app_url : str, optional
        The URL for the Jaqpot application. If not provided, it is derived from the base URL.
    login_url : str, optional
        The URL for the Jaqpot login. If not provided, it is derived from the base URL.
    api_url : str, optional
        The URL for the Jaqpot API. If not provided, it is derived from the base URL.
    keycloak_realm : str, optional
        The Keycloak realm name. Default is "jaqpot".
    keycloak_client_id : str, optional
        The Keycloak client ID. Default is "jaqpot-client".
    create_logs : bool, optional
        Whether to create logs. Default is False.
    """

    def __init__(
        self,
        base_url=None,
        app_url=None,
        login_url=None,
        api_url=None,
        keycloak_realm=None,
        keycloak_client_id=None,
        create_logs=False,
    ):
        # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        self.log = init_logger(
            __name__, testing_mode=False, output_log_file=create_logs
        )
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = "https://jaqpot.org"
        self.app_url = app_url or add_subdomain(self.base_url, "app")
        self.login_url = login_url or add_subdomain(self.base_url, "login")
        self.api_url = api_url or add_subdomain(self.base_url, "api")
        self.keycloak_realm = keycloak_realm or "jaqpot"
        self.keycloak_client_id = keycloak_client_id or "jaqpot-client"
        self.access_token = None
        self.http_client = None

    def _deploy_model(
        self,
        body_model: Model,
        raw_model_bytes: bytes,
        raw_preprocessor_bytes: bytes = None,
    ):
        model_api = ModelApi(self.http_client)
        model_size = len(raw_model_bytes)

        if model_size <= MAX_INLINE_MODEL_SIZE:
            body_model.raw_model = model_to_b64encoding(raw_model_bytes)
            if raw_preprocessor_bytes:
                body_model.raw_preprocessor = model_to_b64encoding(
                    raw_preprocessor_bytes
                )

            self._create_model_request(body_model, model_api)
        else:
            self.log.info("Large model detected, using /v1/large-models flow")

            # Large upload
            large_model_api = LargeModelApi(self.http_client)
            body_model.raw_model = None
            if raw_preprocessor_bytes:
                body_model.raw_preprocessor = model_to_b64encoding(
                    raw_preprocessor_bytes
                )

            response = large_model_api.create_large_model_with_http_info(
                model=body_model
            )

            if response.status_code < 300:
                result = response.data
                model_id = result.model_id
                upload_url = result.upload_url
                self.log.info(
                    "Uploading your model directly to s3, due to its large size..."
                )
                upload_file_to_s3_presigned_url(upload_url, raw_model_bytes)

                response = large_model_api.confirm_large_model_upload_with_http_info(
                    model_id
                )
                if response.status_code < 300:
                    self.log.info(
                        "Model has been successfully uploaded. The url of the model is %s",
                        self.app_url + "/dashboard/models/" + model_id,
                    )
            else:
                self.log.error("Error code: " + str(response.status_code.value))

    def login(self):
        """
        Log in to Jaqpot using Keycloak.

        This method opens a browser window for the user to log in via Keycloak,
        then exchanges the authorization code for an access token.

        Returns
        -------
        None
        """
        # Configure Keycloak client
        keycloak_openid = KeycloakOpenID(
            server_url=self.login_url,
            client_id=self.keycloak_client_id,
            realm_name=self.keycloak_realm,
        )

        # Generate the authorization URL
        redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
        auth_url = keycloak_openid.auth_url(
            redirect_uri=redirect_uri,
            scope="openid email profile",
            state="random_state_value",
        )

        print(f"Open this URL in your browser and log in:\n{auth_url}")

        # Automatically open the browser (optional)
        webbrowser.open(auth_url)

        code = input("Enter the authorization code you received: ")

        # Exchange the code for an access token
        token_response = keycloak_openid.token(
            grant_type="authorization_code", code=code, redirect_uri=redirect_uri
        )

        access_token = token_response["access_token"]
        self.access_token = access_token
        self.http_client = (
            JaqpotApiHttpClientBuilder(host=self.api_url)
            .build_with_access_token(self.access_token)
            .build()
        )

    def _create_model_request(self, body_model, model_api):
        response = model_api.create_model_with_http_info(model=body_model)
        if response.status_code < 300:
            model_url = response.headers.get("Location")
            model_id = model_url.split("/")[-1]

            self.log.info(
                "Model has been successfully uploaded. The url of the model is %s",
                self.app_url + "/dashboard/models/" + model_id,
            )
        else:
            # error = response.headers.get("error")
            # error_description = response.headers.get("error_description")
            self.log.error("Error code: " + str(response.status_code.value))

    def deploy_sklearn_model(self, model, name, description, visibility):
        """
        Deploy an sklearn model on Jaqpot.

        Parameters
        ----------
        model : object
            The sklearn model to be deployed. The model should have attributes
            like `onnx_model`, `onnx_preprocessor`, `type`, `jaqpotpy_version`,
            `doa_data`, `libraries`, `dependentFeatures`, `independentFeatures`,
            `selected_features`, `featurizers`, `preprocessors`, and `scores`.
        name : str
            The name of the model.
        description : str
            A description of the model.
        visibility : str
            The visibility of the model (e.g., 'public', 'private').

        Returns
        -------
        None
        """
        raw_model_bytes = model.onnx_model.SerializeToString()
        raw_preprocessor_bytes = (
            model.onnx_preprocessor.SerializeToString()
            if model.onnx_preprocessor
            else None
        )
        body_model = Model(
            name=name,
            type=model.type,
            jaqpotpy_version=model.jaqpotpy_version,
            doas=model.doa_data,
            libraries=model.libraries,
            dependent_features=[
                Feature(
                    key=feature_i["key"],
                    name=feature_i["name"],
                    feature_type=feature_i["featureType"],
                )
                for feature_i in model.dependentFeatures
            ],
            independent_features=[
                Feature(
                    key=feature_i["key"],
                    name=feature_i["name"],
                    feature_type=feature_i["featureType"],
                    possible_values=feature_i["possible_values"]
                    if "possible_values" in feature_i
                    else None,
                )
                for feature_i in model.independentFeatures
            ],
            visibility=ModelVisibility(visibility),
            task=ModelTask(model.task.upper()),
            selected_features=model.selected_features,
            description=description,
            featurizers=model.featurizers,
            preprocessors=model.preprocessors,
            scores=model.scores,
        )
        self._deploy_model(body_model, raw_model_bytes, raw_preprocessor_bytes)

    def deploy_torch_model(
        self,
        onnx_model,
        featurizer,
        name,
        description,
        target_name,
        visibility,
        task,
    ):
        """
        Deploy a PyTorch model on Jaqpot.

        Parameters
        ----------
        onnx_model : object
            The ONNX model to be deployed.
        featurizer : object
            The featurizer used for preprocessing. The featurizer should have a method `get_dict()`.
        name : str
            The name of the model.
        description : str
            A description of the model.
        target_name : str
            The name of the target feature.
        visibility : str
            The visibility of the model (e.g., 'public', 'private').
        task : str
            The task type (e.g., 'binary_classification', 'regression', 'multiclass_classification').

        Returns
        -------
        None
        """
        # Task
        if task == "binary_classification":
            model_task = ModelTask.BINARY_CLASSIFICATION
            feature_type = FeatureType.INTEGER
        elif task == "regression":
            model_task = ModelTask.REGRESSION
            feature_type = FeatureType.FLOAT
        elif task == "multiclass_classification":
            model_task = ModelTask.MULTICLASS_CLASSIFICATION
            feature_type = FeatureType.INTEGER
        else:
            raise ValueError("Task should be either classification or regression")
        # Type
        if featurizer.__class__.__name__ == "SmilesVectorizer":
            model_type = ModelType.TORCH_SEQUENCE_ONNX
        elif featurizer.__class__.__name__ == "SmilesGraphFeaturizer":
            model_type = ModelType.TORCH_GEOMETRIC_ONNX
        else:
            raise ValueError(
                "Featurizer should be either SmilesVectorizer or SmilesGraphFeaturizer"
            )
        model_api = ModelApi(self.http_client)
        # Change Base URL when not in local testing
        # baseurl: "http://localhost.jaqpot.org:8080/"
        torch_config = featurizer.get_dict()
        body_model = Model(
            name=name,
            type=model_type,
            jaqpotpy_version=jaqpotpy.__version__,
            libraries=get_installed_libraries(),
            dependent_features=[
                Feature(key=target_name, name=target_name, feature_type=feature_type)
            ],  # TODO: Spaces dont work in endpoint name
            independent_features=[
                Feature(key="SMILES", name="SMILES", feature_type=FeatureType.SMILES)
            ],
            torch_config=torch_config,
            task=model_task,
            visibility=ModelVisibility(visibility),
            raw_model=onnx_model,
            description=description,
        )
        response = model_api.create_model_with_http_info(model=body_model)
        if response.status_code < 300:
            model_url = response.headers.get("Location")
            model_id = model_url.split("/")[-1]

            self.log.info(
                "Model has been successfully uploaded. The url of the model is %s",
                self.app_url + "/dashboard/models/" + model_id,
            )
        else:
            self.log.error("Error code: " + str(response.status_code.value))
            self.log.error("Error message: " + response.content.decode("utf-8"))

    def deploy_torch_onnx_model(
        self,
        model: TorchONNXModel,
        name: str,
        description: str,
        visibility: ModelVisibility,
    ):
        """
        Deploy a Torch model on Jaqpot.

        Parameters
        ----------
        model : TorchONNXModel
            The model to be deployed. Must have ONNX and metadata.
        name : str
            The name of the model.
        description : str
            Description of the model.
        visibility : str
            'public' or 'private'

        Returns
        -------
        None
        """
        raw_model_bytes = model.onnx_bytes
        raw_preprocessor_bytes = (
            model.onnx_preprocessor if model.onnx_preprocessor else None
        )

        body_model = Model(
            name=name,
            type=ModelType.TORCH_ONNX,
            libraries=[],
            jaqpotpy_version=jaqpotpy.__version__,
            dependent_features=model.dependent_features,
            independent_features=model.independent_features,
            visibility=visibility,
            task=model.task,
            description=description,
        )

        self._deploy_model(body_model, raw_model_bytes, raw_preprocessor_bytes)

    def deploy_docker_model(
        self,
        model: DockerModel,
        name: str,
        description: str,
        visibility: ModelVisibility,
    ) -> None:
        """
        Deploys a Docker-based model on Jaqpot.

        This method registers a Docker-encapsulated machine learning model on the Jaqpot platform,
        allowing it to be accessed via the Jaqpot API.

        Args:
            model (DockerModel):
                The Docker-based model to be deployed. The model must contain attributes such as:
                - `jaqpotpy_version` (str): The version of JaqpotPy used.
                - `docker_config` (DockerConfig): The Docker container configuration.
                - `dependent_features` (List[Feature]): The output features of the model.
                - `independent_features` (List[Feature]): The input features of the model.

            name (str):
                The name of the model to be displayed on Jaqpot.

            description (str):
                A short textual description of the model.

            visibility (ModelVisibility):
                The access level of the model (e.g., public, private, or restricted to an organization).

        Returns:
            None

        Raises:
            ValueError: If any of the required parameters are invalid or missing.
            JaqpotAPIError: If the deployment request fails due to an API issue.
        """
        model_api = ModelApi(self.http_client)
        raw_model = ""

        body_model = Model(
            name=name,
            type=ModelType.DOCKER,
            jaqpotpy_version=model.jaqpotpy_version,
            dependent_features=[
                Feature(
                    key=feature_i["key"],
                    name=feature_i["name"],
                    feature_type=feature_i["featureType"],
                )
                for feature_i in model.dependent_features
            ],
            independent_features=[
                Feature(
                    key=feature_i["key"],
                    name=feature_i["name"],
                    feature_type=feature_i["featureType"],
                    possible_values=feature_i.get("possible_values"),
                )
                for feature_i in model.independent_features
            ],
            visibility=visibility,
            task=ModelTask.REGRESSION,
            raw_model=raw_model,
            description=description,
            docker_config=model.docker_config.to_dict(),
            libraries=[],
        )

        self._create_model_request(body_model, model_api)
