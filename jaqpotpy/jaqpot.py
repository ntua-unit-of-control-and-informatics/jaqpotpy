import webbrowser

from keycloak import KeycloakOpenID

import jaqpotpy
from jaqpotpy.api.get_installed_libraries import get_installed_libraries
from jaqpotpy.api.jaqpot_api_client_builder import JaqpotApiHttpClientBuilder
from jaqpotpy.api.jaqpot_api_http_client import JaqpotApiHttpClient
from jaqpotpy.api.model_to_b64encoding import model_to_b64encoding
from jaqpotpy.api.openapi.api.model_api import ModelApi
from jaqpotpy.api.openapi.models.feature import Feature
from jaqpotpy.api.openapi.models.feature_type import FeatureType
from jaqpotpy.api.openapi.models.model import Model
from jaqpotpy.api.openapi.models.model_task import ModelTask
from jaqpotpy.api.openapi.models.model_type import ModelType
from jaqpotpy.api.openapi.models.model_visibility import ModelVisibility
from jaqpotpy.helpers.logging import init_logger
from jaqpotpy.utils.url_utils import add_subdomain

ENCODING = "utf-8"


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
            self.base_url = "https://jaqpot.org"
        self.app_url = app_url or add_subdomain(self.base_url, "app")
        self.login_url = login_url or add_subdomain(self.base_url, "login")
        self.api_url = api_url or add_subdomain(self.base_url, "api")
        self.keycloak_realm = keycloak_realm or "jaqpot"
        self.keycloak_client_id = keycloak_client_id or "jaqpot-client"
        self.access_token = None
        self.http_client = None

    def login(self):
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

    def deploy_sklearn_model(self, model, name, description, visibility):
        """ "
        Deploy sklearn models on Jaqpot.
        :param model:
        :param name:
        :param description:
        :param visibility:
        :return:
        """
        model_api = ModelApi(self.http_client)
        raw_model = model_to_b64encoding(model.onnx_model.SerializeToString())
        raw_preprocessor = (
            model_to_b64encoding(model.onnx_preprocessor.SerializeToString())
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
            raw_preprocessor=raw_preprocessor,
            raw_model=raw_model,
            selected_features=model.selected_features,
            description=description,
            featurizers=model.featurizers,
            preprocessors=model.preprocessors,
            scores=model.scores,
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
            # error = response.headers.get("error")
            # error_description = response.headers.get("error_description")
            self.log.error("Error code: " + str(response.status_code.value))

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
        model_api = ModelApi(self.http_client)
        # Change Base URL when not in local testing
        # baseurl: "http://localhost.jaqpot.org:8080/"
        featurizer_dict = featurizer.get_dict()

        featurizer_config = featurizer_dict
        torch_config_json = {"featurizerConfig": featurizer_config}
        torch_config = torch_config_json
        body_model = Model(
            name=name,
            type=ModelType.TORCH_ONNX,
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
            self.log.info(
                "Graph Pytorch Model has been successfully uploaded. The url of the model is "
                + response.headers.get("Location")
            )
        else:
            self.log.error("Error code: " + str(response.status_code.value))
            self.log.error("Error message: " + response.content.decode("utf-8"))
