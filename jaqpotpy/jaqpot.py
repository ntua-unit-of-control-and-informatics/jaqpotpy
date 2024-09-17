import http.client as http_client
import json
import time
import webbrowser

import numpy as np
import pandas as pd
import jaqpotpy
from keycloak import KeycloakOpenID
import jaqpotpy.api.dataset_api as data_api
import jaqpotpy.api.doa_api as doa_api
import jaqpotpy.api.feature_api as featapi
import jaqpotpy.api.models_api as models_api
from jaqpotpy.api.openapi.jaqpot_api_client.models.model_extra_config import (
    ModelExtraConfig,
)
from jaqpotpy.api.openapi.jaqpot_api_client.models.model_extra_config_torch_config import (
    ModelExtraConfigTorchConfig,
)
from jaqpotpy.api.openapi.jaqpot_api_client.models.model_extra_config_torch_config_additional_property import (
    ModelExtraConfigTorchConfigAdditionalProperty,
)
import jaqpotpy.api.task_api as task_api
import jaqpotpy.doa.doa as jha
import jaqpotpy.helpers.dataset_deserializer as ds
from jaqpotpy.api.openapi.jaqpot_api_client.api.model import create_model
from jaqpotpy.api.openapi.jaqpot_api_client.models import Model
from jaqpotpy.api.openapi.jaqpot_api_client.models.model_type import ModelType
from jaqpotpy.api.openapi.jaqpot_api_client.models.model_task import ModelTask
from jaqpotpy.api.openapi.jaqpot_api_client.models.model_visibility import (
    ModelVisibility,
)
from jaqpotpy.api.openapi.jaqpot_api_client.models.feature import Feature
from jaqpotpy.api.openapi.jaqpot_api_client.models.feature_type import FeatureType
from jaqpotpy.api.openapi.jaqpot_api_client.client import AuthenticatedClient
from jaqpotpy.api.model_to_b64encoding import model_to_b64encoding
from jaqpotpy.helpers.logging import init_logger
from jaqpotpy.helpers.serializer import JaqpotSerializer
from jaqpotpy.utils.url_utils import add_subdomain
from jaqpotpy.api.get_installed_libraries import get_installed_libraries

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

    def __init__(self, base_url=None, create_logs=False):
        # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        self.log = init_logger(
            __name__, testing_mode=False, output_log_file=create_logs
        )
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = "https://appv2.jaqpot.org/"
        self.app_url = add_subdomain(self.base_url, "app")
        self.login_url = add_subdomain(self.base_url, "login")
        self.api_url = add_subdomain(self.base_url, "api")
        self.api_key = None
        self.user_id = None
        self.http_client = http_client

    def login(self):
        """Logins on Jaqpot."""
        try:
            # Configure Keycloak client
            keycloak_openid = KeycloakOpenID(
                server_url=self.login_url,
                client_id="jaqpot-client",
                realm_name="jaqpot",
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
        except Exception:
            self.log.error("Could not login to jaqpot")

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
        actual_model = model_to_b64encoding(model.copy().onnx_model.SerializeToString())
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

    def deploy_Torch_Graph_model(
        self, onnx_model, featurizer, name, description, target_name, visibility, task
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
        body_model = Model(
            name=name,
            type=ModelType.TORCH,
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
            # error = response.headers.get("error")
            # error_description = response.headers.get("error_description")
            self.log.error("Error code: " + str(response.status_code.value))

    def deploy_XGBoost(self, model, X, y, title, description, algorithm, doa=None):
        """Deploys XGBoost model to Jaqpot.

        Extended description of function.

        Parameters
        ----------
        model : XGBoost trained model
            model is a trained model that occurs from the XGBoost family of algorithms
        X : pandas dataframe
            The dataframe that is used to train the model (X variables).
        y : pandas dataframe
            The dataframe that is used to train the model (y variables).
        title: String
            The title of the model
        description: String
            The description of the model
        algorithm: String
            The algorithm that the model implements
        doa: pandas dataframe
            The dataset used to create the domain of applicability of the model

        Returns
        -------
        string
            The id of the model that uploaded

        """
        if isinstance(X, pd.DataFrame) is False:
            raise Exception(
                "Function deploy_glm supports pandas dataframe or series. X is not one"
            )
        if isinstance(y, pd.DataFrame) is False and isinstance(y, pd.Series) is False:
            raise Exception(
                "Function deploy_glm supports pandas dataframe or series. Y is not one"
            )
        # coef_flatten = model.coef_.flatten()
        # intercept_flatten = model.intercept_.flatten()
        # coef_df = pd.DataFrame(list(zip(list(X), coef_flatten)), columns=['Features', 'coeff'])
        # intercept = pd.DataFrame([('(intercept)', intercept_flatten[0])], columns=['Features', 'coeff'])
        # coefs_all = intercept.append(coef_df, ignore_index=True)
        # coefs = {}
        additionalInfo = {}
        # for key in coefs_all.values:
        #     coefs[key[0]] = key[1]
        # additionalInfo['coefficients'] = coefs
        additionalInfo["inputSeries"] = list(X)
        pretrained = help.create_pretrain_req(
            model,
            X,
            y,
            title,
            description,
            algorithm,
            "XGBoost model",
            "XGBoost-model",
            additionalInfo,
        )
        # j = json.dumps(pretrained)
        j = json.dumps(pretrained, cls=JaqpotSerializer)
        if doa is None:
            response = models_api.post_pretrained_model(
                self.base_url, self.api_key, j, self.log
            )
            if response.status_code < 300:
                resp = response.json()
                self.log.info(
                    "Model with id: "
                    + resp["modelId"]
                    + " created. Please visit the application to proceed"
                )
                return resp["modelId"]
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp["message"])
                return
        else:
            response = models_api.post_pretrained_model(
                self.base_url, self.api_key, j, self.log
            )
            if response.status_code < 300:
                resp = response.json()
                self.log.info(
                    "Model with id: "
                    + resp["modelId"]
                    + " created. Storing Domain of applicability"
                )
                modid = resp["modelId"]
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp["message"])
                return
            # loop = asyncio.get_event_loop()
            # a = loop.create_task(jha.calculate_a(X))
            # b = loop.create_task(jha.calculate_doa_matrix(X))

            a = jha.calculate_a(X)
            b = jha.calculate_doa_matrix(X)
            # all_groups = asyncio.gather(a, b)
            # results = loop.run_until_complete(all_groups)
            doa = help.create_doa(inv_m=b.values.tolist(), a=a, modelid=resp["modelId"])
            j = json.dumps(doa, cls=JaqpotSerializer)
            resp = doa_api.post_models_doa(self.base_url, self.api_key, j, self.log)
            if resp == 201:
                self.log.info(
                    "Stored Domain of applicability. Visit the application to proceed"
                )
                return modid

    def get_model_by_id(self, model):
        """Retrieves user's model by ID.

        Parameters
        ----------
        model : str
            The model's ID.


        Returns
        -------
        Object
            The particular model.

        """
        return models_api.get_model(self.base_url, self.api_key, model, self.log)

    def get_feature_by_id(self, feature):
        """Retrieves a Jaqpot feature.

        Parameters
        ----------
        feature : str
            The feature's ID.


        Returns
        -------
        Object
            The particular feature.

        """
        return featapi.get_feature(self.base_url, self.api_key, feature)

    def get_my_models(self, minimum, maximum):
        """Retrieves user's models.

        Parameters
        ----------
        minimum : int
            The index of the first model.
        maximum : int
            The index of the last model.

        Returns
        -------
        Object
            The models of the user.

        """
        return models_api.get_my_models(
            self.base_url, self.api_key, minimum, maximum, self.log
        )

    def get_orgs_models(self, organization, minimum, maximum):
        """Retrieves organization's models.

        Parameters
        ----------
        organization: str
            The organization's ID.
        minimum : int
            The index of the first model.
        maximum : int
            The index of the last model.

        Returns
        -------
        Object
            The models of the organization.

        """
        return models_api.get_orgs_models(
            self.base_url, self.api_key, organization, minimum, maximum, self.log
        )

    def get_models_by_tag(self, tag, minimum, maximum):
        """Retrieves models with a particular tag.

        Parameters
        ----------
        tag: str
            The model's tag.
        minimum : int
            The index of the first model.
        maximum : int
            The index of the last model.

        Returns
        -------
        Object
            The models of the organization.

        """
        return models_api.get_models_by_tag(
            self.base_url, self.api_key, tag, minimum, maximum, self.log
        )

    def get_models_by_tag_and_org(self, organization, tag, minimum, maximum):
        """Retrieves models of an organization with a particular tag.

        Parameters
        ----------
        organization: str:
            The organization's ID.
        tag: str
            The model's tag.
        minimum : int
            The index of the first model.
        maximum : int
            The index of the last model.

        Returns
        -------
        Object
            The models of the organization.

        """
        return models_api.get_models_by_tag(
            self.base_url, self.api_key, organization, tag, minimum, maximum, self.log
        )

    def get_dataset(self, dataset):
        """Retrieves a dataset.

        Parameters
        ----------
        dataset: str
            The dataset's ID.

        Returns
        -------
        pd.DataFrame()
            DataFrame of the dataset.

        Note
        ----
        Mazimum rows retrieved: 1000.

        """
        dataRetrieved = data_api.get_dataset(
            self.base_url, self.api_key, dataset, self.log
        )

        feats = ["" for i in range(len(dataRetrieved["features"]))]

        for item in dataRetrieved["features"]:
            feats[int(item["key"])] = item["name"]

        indices = [item["entryId"]["name"] for item in dataRetrieved["dataEntry"]]

        data = [
            [item["values"][str(feats.index(col))] for col in feats]
            for item in dataRetrieved["dataEntry"]
        ]

        return pd.DataFrame(data, index=indices, columns=feats)

    def get_doa(self, model):
        """Retrieves model's domain of applicability.

        Parameters
        ----------
        model: str
            The model's ID.

        Returns
        -------
        Object
            The model's domain of applicability.

        """
        return doa_api.get_models_doa(self.base_url, self.api_key, model, self.log)
