import http.client as http_client
import webbrowser

from keycloak import KeycloakOpenID

import jaqpotpy
from jaqpotpy.api.get_installed_libraries import get_installed_libraries
from jaqpotpy.api.model_to_b64encoding import model_to_b64encoding
from jaqpotpy.api.openapi.jaqpot_api_client.types import UNSET
from jaqpotpy.api.openapi.jaqpot_api_client.api.model import create_model
from jaqpotpy.api.openapi.jaqpot_api_client.client import AuthenticatedClient
from jaqpotpy.api.openapi.jaqpot_api_client.models import Model
from jaqpotpy.api.openapi.jaqpot_api_client.models.feature import Feature
from jaqpotpy.api.openapi.jaqpot_api_client.models.feature_type import FeatureType
from jaqpotpy.api.openapi.jaqpot_api_client.models.model_extra_config import (
    ModelExtraConfig,
)
from jaqpotpy.api.openapi.jaqpot_api_client.models.model_extra_config_torch_config import (
    ModelExtraConfigTorchConfig,
)
from jaqpotpy.api.openapi.jaqpot_api_client.models.model_extra_config_torch_config_additional_property import (
    ModelExtraConfigTorchConfigAdditionalProperty,
)
from jaqpotpy.api.openapi.jaqpot_api_client.models.model_task import ModelTask
from jaqpotpy.api.openapi.jaqpot_api_client.models.model_type import ModelType
from jaqpotpy.api.openapi.jaqpot_api_client.models.model_visibility import (
    ModelVisibility,
)
from jaqpotpy.helpers.logging import init_logger
from jaqpotpy.utils.url_utils import add_subdomain

ENCODING = "utf-8"


# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


class Jaqpot:
    """Deploys sklearn models on Jaqpot.

    Extended description of function.

    Parameters
    ----------
    base_url : The url on which Jaqpot services are deployed

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
            self.base_url = "https://appv2.jaqpot.org/"
        self.app_url = app_url or add_subdomain(self.base_url, "app")
        self.login_url = login_url or add_subdomain(self.base_url, "login")
        self.api_url = api_url or add_subdomain(self.base_url, "api")
        self.keycloak_realm = keycloak_realm or "jaqpot"
        self.keycloak_client_id = keycloak_client_id or "jaqpot-client"
        self.api_key = None
        self.user_id = None
        self.http_client = http_client

    def login(self):
        """Logins on Jaqpot."""
        try:
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
            self.api_key = access_token
        except Exception as ex:
            self.log.error("Could not login to jaqpot", exc_info=ex)

    def set_api_key(self, api_key):
        """Set's api key for authentication on Jaqpot.

        Parameters
        ----------
        api_key : api_key can be retireved from the application after logged in

        """
        self.api_key = api_key
        self.log.info("api key is set")

    def deploy_sklearn_model(self, model, name, description, visibility):
        """ "
        Deploy sklearn models on Jaqpot.
        :param model:
        :param name:
        :param description:
        :param visibility:
        :return:
        """
        auth_client = AuthenticatedClient(base_url=self.api_url, token=self.api_key)
        actual_model = model_to_b64encoding(model.onnx_model.SerializeToString())
        body_model = Model(
            name=name,
            type=model.type,
            jaqpotpy_version=model.jaqpotpy_version,
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
                    else UNSET,
                )
                for feature_i in model.independentFeatures
            ],
            visibility=ModelVisibility(visibility),
            task=ModelTask(model.task.upper()),
            actual_model=actual_model,
            description=description,
            extra_config=model.extra_config,
        )
        response = create_model.sync_detailed(client=auth_client, body=body_model)
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

    def deploy_torch_model(
        self,
        onnx_model,
        type,
        featurizer,
        name,
        description,
        target_name,
        visibility,
        task,
    ):
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
        auth_client = AuthenticatedClient(
            base_url=self.api_url, token=self.api_key
        )  # Change Base URL when not in local testing
        # baseurl: "http://localhost.jaqpot.org:8080/"
        featurizer_dict = featurizer.get_dict()

        featurizer_config = ModelExtraConfigTorchConfigAdditionalProperty.from_dict(
            featurizer_dict
        )
        torch_config_json = {"featurizerConfig": featurizer_config.to_dict()}
        torch_config = ModelExtraConfigTorchConfig.from_dict(torch_config_json)
        if type == "TORCHSCRIPT":
            type = ModelType.TORCHSCRIPT
        elif type == "TORCH_ONNX":
            type = ModelType.TORCH_ONNX
        body_model = Model(
            name=name,
            type=type,
            jaqpotpy_version=jaqpotpy.__version__,
            libraries=get_installed_libraries(),
            dependent_features=[
                Feature(key=target_name, name=target_name, feature_type=feature_type)
            ],  # TODO: Spaces dont work in endpoint name
            independent_features=[
                Feature(key="SMILES", name="SMILES", feature_type=FeatureType.SMILES)
            ],
            extra_config=ModelExtraConfig(torch_config=torch_config),
            task=model_task,
            visibility=ModelVisibility(visibility),
            actual_model=onnx_model,
            description=description,
        )
        response = create_model.sync_detailed(client=auth_client, body=body_model)
        if response.status_code < 300:
            self.log.info(
                "Graph Pytorch Model has been successfully uploaded. The url of the model is "
                + response.headers.get("Location")
            )
        else:
            self.log.error("Error code: " + str(response.status_code.value))
            self.log.error("Error message: " + response.content.decode("utf-8"))
