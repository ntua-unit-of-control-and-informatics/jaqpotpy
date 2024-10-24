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

from jaqpotpy.api.openapi.api.model_api import ModelApi


class TestModelApi(unittest.TestCase):
    """ModelApi unit test stubs"""

    def setUp(self) -> None:
        self.api = ModelApi()

    def tearDown(self) -> None:
        pass

    def test_create_model(self) -> None:
        """Test case for create_model

        Create a new model
        """
        pass

    def test_delete_model_by_id(self) -> None:
        """Test case for delete_model_by_id

        Delete a Model
        """
        pass

    def test_get_legacy_model_by_id(self) -> None:
        """Test case for get_legacy_model_by_id

        Get a legacy model
        """
        pass

    def test_get_model_by_id(self) -> None:
        """Test case for get_model_by_id

        Get a Model
        """
        pass

    def test_get_models(self) -> None:
        """Test case for get_models

        Get paginated models
        """
        pass

    def test_get_shared_models(self) -> None:
        """Test case for get_shared_models

        Get paginated shared models
        """
        pass

    def test_partially_update_model(self) -> None:
        """Test case for partially_update_model

        Partially update specific fields of a model
        """
        pass

    def test_predict_with_model(self) -> None:
        """Test case for predict_with_model

        Predict with Model
        """
        pass

    def test_predict_with_model_csv(self) -> None:
        """Test case for predict_with_model_csv

        Predict using CSV with Model
        """
        pass

    def test_search_models(self) -> None:
        """Test case for search_models

        Search for models
        """
        pass


if __name__ == '__main__':
    unittest.main()
