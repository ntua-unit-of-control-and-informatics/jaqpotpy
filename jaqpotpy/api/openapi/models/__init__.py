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


# import models into model package
from jaqpotpy.api.openapi.models.create_invitations_request import CreateInvitationsRequest
from jaqpotpy.api.openapi.models.dataset import Dataset
from jaqpotpy.api.openapi.models.dataset_csv import DatasetCSV
from jaqpotpy.api.openapi.models.dataset_type import DatasetType
from jaqpotpy.api.openapi.models.error_code import ErrorCode
from jaqpotpy.api.openapi.models.error_response import ErrorResponse
from jaqpotpy.api.openapi.models.feature import Feature
from jaqpotpy.api.openapi.models.feature_possible_value import FeaturePossibleValue
from jaqpotpy.api.openapi.models.feature_type import FeatureType
from jaqpotpy.api.openapi.models.get_datasets200_response import GetDatasets200Response
from jaqpotpy.api.openapi.models.get_models200_response import GetModels200Response
from jaqpotpy.api.openapi.models.lead import Lead
from jaqpotpy.api.openapi.models.library import Library
from jaqpotpy.api.openapi.models.model import Model
from jaqpotpy.api.openapi.models.model_extra_config import ModelExtraConfig
from jaqpotpy.api.openapi.models.model_summary import ModelSummary
from jaqpotpy.api.openapi.models.model_task import ModelTask
from jaqpotpy.api.openapi.models.model_type import ModelType
from jaqpotpy.api.openapi.models.model_visibility import ModelVisibility
from jaqpotpy.api.openapi.models.organization import Organization
from jaqpotpy.api.openapi.models.organization_invitation import OrganizationInvitation
from jaqpotpy.api.openapi.models.organization_summary import OrganizationSummary
from jaqpotpy.api.openapi.models.organization_user import OrganizationUser
from jaqpotpy.api.openapi.models.organization_user_association_type import OrganizationUserAssociationType
from jaqpotpy.api.openapi.models.organization_visibility import OrganizationVisibility
from jaqpotpy.api.openapi.models.partial_update_organization_request import PartialUpdateOrganizationRequest
from jaqpotpy.api.openapi.models.partially_update_model_feature_request import PartiallyUpdateModelFeatureRequest
from jaqpotpy.api.openapi.models.partially_update_model_request import PartiallyUpdateModelRequest
from jaqpotpy.api.openapi.models.transformer import Transformer
from jaqpotpy.api.openapi.models.user import User