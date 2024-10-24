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

from jaqpotpy.api.openapi.models.dataset import Dataset

class TestDataset(unittest.TestCase):
    """Dataset unit test stubs"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def make_instance(self, include_optional) -> Dataset:
        """Test Dataset
            include_optional is a boolean, when False only required
            params are included, when True both required and
            optional params are included """
        # uncomment below to create an instance of `Dataset`
        """
        model = Dataset()
        if include_optional:
            return Dataset(
                id = 1,
                type = 'PREDICTION',
                entry_type = 'ARRAY',
                input = [
                    null
                    ],
                result = [
                    null
                    ],
                status = 'CREATED',
                failure_reason = '',
                user_id = '',
                model_id = 56,
                model_name = '',
                executed_at = '',
                execution_finished_at = '',
                created_at = '',
                updated_at = ''
            )
        else:
            return Dataset(
                type = 'PREDICTION',
                entry_type = 'ARRAY',
                input = [
                    null
                    ],
        )
        """

    def testDataset(self):
        """Test Dataset"""
        # inst_req_only = self.make_instance(include_optional=False)
        # inst_req_and_optional = self.make_instance(include_optional=True)

if __name__ == '__main__':
    unittest.main()
