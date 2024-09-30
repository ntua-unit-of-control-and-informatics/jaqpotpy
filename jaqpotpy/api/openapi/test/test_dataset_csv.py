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

from jaqpotpy.api.openapi.models.dataset_csv import DatasetCSV

class TestDatasetCSV(unittest.TestCase):
    """DatasetCSV unit test stubs"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def make_instance(self, include_optional) -> DatasetCSV:
        """Test DatasetCSV
            include_optional is a boolean, when False only required
            params are included, when True both required and
            optional params are included """
        # uncomment below to create an instance of `DatasetCSV`
        """
        model = DatasetCSV()
        if include_optional:
            return DatasetCSV(
                id = 1,
                type = 'PREDICTION',
                input_file = 'YQ==',
                values = [
                    null
                    ],
                status = 'CREATED',
                failure_reason = '',
                model_id = 56,
                model_name = '',
                executed_at = '',
                execution_finished_at = '',
                created_at = '',
                updated_at = ''
            )
        else:
            return DatasetCSV(
                type = 'PREDICTION',
                input_file = 'YQ==',
        )
        """

    def testDatasetCSV(self):
        """Test DatasetCSV"""
        # inst_req_only = self.make_instance(include_optional=False)
        # inst_req_and_optional = self.make_instance(include_optional=True)

if __name__ == '__main__':
    unittest.main()
