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

from jaqpotpy.api.openapi.models.partially_update_model_request import PartiallyUpdateModelRequest

class TestPartiallyUpdateModelRequest(unittest.TestCase):
    """PartiallyUpdateModelRequest unit test stubs"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def make_instance(self, include_optional) -> PartiallyUpdateModelRequest:
        """Test PartiallyUpdateModelRequest
            include_optional is a boolean, when False only required
            params are included, when True both required and
            optional params are included """
        # uncomment below to create an instance of `PartiallyUpdateModelRequest`
        """
        model = PartiallyUpdateModelRequest()
        if include_optional:
            return PartiallyUpdateModelRequest(
                name = '012',
                description = '012',
                visibility = 'PUBLIC',
                task = 'REGRESSION',
                tags = '',
                shared_with_organization_ids = [
                    56
                    ]
            )
        else:
            return PartiallyUpdateModelRequest(
                name = '012',
                visibility = 'PUBLIC',
                task = 'REGRESSION',
        )
        """

    def testPartiallyUpdateModelRequest(self):
        """Test PartiallyUpdateModelRequest"""
        # inst_req_only = self.make_instance(include_optional=False)
        # inst_req_and_optional = self.make_instance(include_optional=True)

if __name__ == '__main__':
    unittest.main()