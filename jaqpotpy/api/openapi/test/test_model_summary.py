# coding: utf-8

"""
    Jaqpot API

    A modern RESTful API for model management and prediction services, built using Spring Boot and Kotlin. Supports seamless integration with machine learning workflows.

    The version of the OpenAPI document: 1.0.0
    Contact: upci.ntua@gmail.com
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""  # noqa: E501


import unittest

from jaqpotpy.api.openapi.models.model_summary import ModelSummary

class TestModelSummary(unittest.TestCase):
    """ModelSummary unit test stubs"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def make_instance(self, include_optional) -> ModelSummary:
        """Test ModelSummary
            include_optional is a boolean, when False only required
            params are included, when True both required and
            optional params are included """
        # uncomment below to create an instance of `ModelSummary`
        """
        model = ModelSummary()
        if include_optional:
            return ModelSummary(
                id = 0,
                name = 'My Model',
                visibility = 'PUBLIC',
                description = 'A description of your model',
                creator = jaqpotpy.api.openapi.models.user.User(
                    id = '', 
                    username = '', 
                    first_name = '', 
                    last_name = '', 
                    email = '', 
                    email_verified = True, ),
                type = 'SKLEARN',
                dependent_features_length = 56,
                independent_features_length = 56,
                shared_with_organizations = [
                    jaqpotpy.api.openapi.models.organization_summary.OrganizationSummary(
                        id = 0, 
                        name = 'My Organization', )
                    ],
                created_at = '2023-01-01T12:00Z',
                updated_at = '2023-01-01T12:00:00Z'
            )
        else:
            return ModelSummary(
                id = 0,
                name = 'My Model',
                visibility = 'PUBLIC',
                type = 'SKLEARN',
                shared_with_organizations = [
                    jaqpotpy.api.openapi.models.organization_summary.OrganizationSummary(
                        id = 0, 
                        name = 'My Organization', )
                    ],
                created_at = '2023-01-01T12:00Z',
        )
        """

    def testModelSummary(self):
        """Test ModelSummary"""
        # inst_req_only = self.make_instance(include_optional=False)
        # inst_req_and_optional = self.make_instance(include_optional=True)

if __name__ == '__main__':
    unittest.main()
