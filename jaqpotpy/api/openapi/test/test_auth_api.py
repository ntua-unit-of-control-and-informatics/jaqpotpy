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

from openapi_client.api.auth_api import AuthApi


class TestAuthApi(unittest.TestCase):
    """AuthApi unit test stubs"""

    def setUp(self) -> None:
        self.api = AuthApi()

    def tearDown(self) -> None:
        pass

    def test_validate_jwt(self) -> None:
        """Test case for validate_jwt

        Validate JWT
        """
        pass


if __name__ == '__main__':
    unittest.main()
