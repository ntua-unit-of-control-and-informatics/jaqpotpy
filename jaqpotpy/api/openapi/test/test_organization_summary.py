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

from jaqpotpy.api.openapi.models.organization_summary import OrganizationSummary

class TestOrganizationSummary(unittest.TestCase):
    """OrganizationSummary unit test stubs"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def make_instance(self, include_optional) -> OrganizationSummary:
        """Test OrganizationSummary
            include_optional is a boolean, when False only required
            params are included, when True both required and
            optional params are included """
        # uncomment below to create an instance of `OrganizationSummary`
        """
        model = OrganizationSummary()
        if include_optional:
            return OrganizationSummary(
                id = 0,
                name = 'My Organization'
            )
        else:
            return OrganizationSummary(
                id = 0,
                name = 'My Organization',
        )
        """

    def testOrganizationSummary(self):
        """Test OrganizationSummary"""
        # inst_req_only = self.make_instance(include_optional=False)
        # inst_req_and_optional = self.make_instance(include_optional=True)

if __name__ == '__main__':
    unittest.main()
