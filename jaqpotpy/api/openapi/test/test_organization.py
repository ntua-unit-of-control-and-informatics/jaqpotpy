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

from jaqpotpy.api.openapi.models.organization import Organization

class TestOrganization(unittest.TestCase):
    """Organization unit test stubs"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def make_instance(self, include_optional) -> Organization:
        """Test Organization
            include_optional is a boolean, when False only required
            params are included, when True both required and
            optional params are included """
        # uncomment below to create an instance of `Organization`
        """
        model = Organization()
        if include_optional:
            return Organization(
                id = 56,
                name = 'my-awesome-org',
                creator = jaqpotpy.api.openapi.models.user.User(
                    id = '', 
                    username = '', 
                    first_name = '', 
                    last_name = '', 
                    email = '', 
                    email_verified = True, ),
                visibility = 'PUBLIC',
                description = 'An awesome organization for managing models.',
                organization_members = [
                    jaqpotpy.api.openapi.models.organization_user.OrganizationUser(
                        id = 56, 
                        user_id = '', 
                        username = '', 
                        email = '', 
                        association_type = 'ADMIN', )
                    ],
                contact_email = 'contact@my-awesome-org.com',
                contact_phone = '+1234567890',
                website = 'http://www.my-awesome-org.com',
                address = '123 Organization St., City, Country',
                can_edit = True,
                is_member = True,
                created_at = '',
                updated_at = ''
            )
        else:
            return Organization(
                name = 'my-awesome-org',
                visibility = 'PUBLIC',
                contact_email = 'contact@my-awesome-org.com',
        )
        """

    def testOrganization(self):
        """Test Organization"""
        # inst_req_only = self.make_instance(include_optional=False)
        # inst_req_and_optional = self.make_instance(include_optional=True)

if __name__ == '__main__':
    unittest.main()
