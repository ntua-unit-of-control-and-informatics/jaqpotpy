# coding: utf-8

# flake8: noqa

"""
    Jaqpot API

    A modern RESTful API for model management and prediction services, built using Spring Boot and Kotlin. Supports seamless integration with machine learning workflows.

    The version of the OpenAPI document: 1.0.0
    Contact: upci.ntua@gmail.com
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""  # noqa: E501


__version__ = "1.0.0"

# import apis into sdk package
from jaqpotpy.api.openapi.api.api_keys_api import ApiKeysApi
from jaqpotpy.api.openapi.api.auth_api import AuthApi
from jaqpotpy.api.openapi.api.dataset_api import DatasetApi
from jaqpotpy.api.openapi.api.feature_api import FeatureApi
from jaqpotpy.api.openapi.api.lead_api import LeadApi
from jaqpotpy.api.openapi.api.model_api import ModelApi
from jaqpotpy.api.openapi.api.organization_api import OrganizationApi
from jaqpotpy.api.openapi.api.organization_invitation_api import OrganizationInvitationApi

# import ApiClient
from jaqpotpy.api.openapi.api_response import ApiResponse
from jaqpotpy.api.openapi.api_client import ApiClient
from jaqpotpy.api.openapi.configuration import Configuration
from jaqpotpy.api.openapi.exceptions import OpenApiException
from jaqpotpy.api.openapi.exceptions import ApiTypeError
from jaqpotpy.api.openapi.exceptions import ApiValueError
from jaqpotpy.api.openapi.exceptions import ApiKeyError
from jaqpotpy.api.openapi.exceptions import ApiAttributeError
from jaqpotpy.api.openapi.exceptions import ApiException

# import models into sdk package
from jaqpotpy.api.openapi.models.api_key import ApiKey
from jaqpotpy.api.openapi.models.binary_classification_scores import BinaryClassificationScores
from jaqpotpy.api.openapi.models.bounding_box_doa import BoundingBoxDoa
from jaqpotpy.api.openapi.models.city_block_doa import CityBlockDoa
from jaqpotpy.api.openapi.models.create_api_key201_response import CreateApiKey201Response
from jaqpotpy.api.openapi.models.create_invitations_request import CreateInvitationsRequest
from jaqpotpy.api.openapi.models.dataset import Dataset
from jaqpotpy.api.openapi.models.dataset_csv import DatasetCSV
from jaqpotpy.api.openapi.models.dataset_type import DatasetType
from jaqpotpy.api.openapi.models.doa import Doa
from jaqpotpy.api.openapi.models.doa_method import DoaMethod
from jaqpotpy.api.openapi.models.error_code import ErrorCode
from jaqpotpy.api.openapi.models.error_response import ErrorResponse
from jaqpotpy.api.openapi.models.feature import Feature
from jaqpotpy.api.openapi.models.feature_possible_value import FeaturePossibleValue
from jaqpotpy.api.openapi.models.feature_type import FeatureType
from jaqpotpy.api.openapi.models.get_all_api_keys_for_user200_response_inner import GetAllApiKeysForUser200ResponseInner
from jaqpotpy.api.openapi.models.get_datasets200_response import GetDatasets200Response
from jaqpotpy.api.openapi.models.get_models200_response import GetModels200Response
from jaqpotpy.api.openapi.models.kernel_based_doa import KernelBasedDoa
from jaqpotpy.api.openapi.models.lead import Lead
from jaqpotpy.api.openapi.models.leverage_doa import LeverageDoa
from jaqpotpy.api.openapi.models.library import Library
from jaqpotpy.api.openapi.models.mahalanobis_doa import MahalanobisDoa
from jaqpotpy.api.openapi.models.mean_var_doa import MeanVarDoa
from jaqpotpy.api.openapi.models.model import Model
from jaqpotpy.api.openapi.models.model_extra_config import ModelExtraConfig
from jaqpotpy.api.openapi.models.model_scores import ModelScores
from jaqpotpy.api.openapi.models.model_summary import ModelSummary
from jaqpotpy.api.openapi.models.model_task import ModelTask
from jaqpotpy.api.openapi.models.model_type import ModelType
from jaqpotpy.api.openapi.models.model_visibility import ModelVisibility
from jaqpotpy.api.openapi.models.multiclass_classification_scores import MulticlassClassificationScores
from jaqpotpy.api.openapi.models.organization import Organization
from jaqpotpy.api.openapi.models.organization_invitation import OrganizationInvitation
from jaqpotpy.api.openapi.models.organization_summary import OrganizationSummary
from jaqpotpy.api.openapi.models.organization_user import OrganizationUser
from jaqpotpy.api.openapi.models.organization_user_association_type import OrganizationUserAssociationType
from jaqpotpy.api.openapi.models.organization_visibility import OrganizationVisibility
from jaqpotpy.api.openapi.models.partial_update_organization_request import PartialUpdateOrganizationRequest
from jaqpotpy.api.openapi.models.partially_update_model_feature_request import PartiallyUpdateModelFeatureRequest
from jaqpotpy.api.openapi.models.partially_update_model_request import PartiallyUpdateModelRequest
from jaqpotpy.api.openapi.models.prediction_doa import PredictionDoa
from jaqpotpy.api.openapi.models.prediction_model import PredictionModel
from jaqpotpy.api.openapi.models.prediction_request import PredictionRequest
from jaqpotpy.api.openapi.models.prediction_response import PredictionResponse
from jaqpotpy.api.openapi.models.regression_scores import RegressionScores
from jaqpotpy.api.openapi.models.scores import Scores
from jaqpotpy.api.openapi.models.transformer import Transformer
from jaqpotpy.api.openapi.models.update_api_key200_response import UpdateApiKey200Response
from jaqpotpy.api.openapi.models.update_api_key_request import UpdateApiKeyRequest
from jaqpotpy.api.openapi.models.user import User
