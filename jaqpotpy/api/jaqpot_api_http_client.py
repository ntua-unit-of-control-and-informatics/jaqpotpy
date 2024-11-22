from jaqpotpy.api.openapi import Configuration, ApiClient


class JaqpotApiHttpClient(ApiClient):
    """
    A client for interacting with the Jaqpot API over HTTP.
    """

    def __init__(self, host=None, access_token=None):
        """
        Initialize the JaqpotApiHttpClient.

        :param host: str, optional
            The base URL of the Jaqpot API. Defaults to "https://api.jaqpot.com".
        :param access_token: str, optional
            The access token for authenticating with the Jaqpot API.
        """
        self.host = host or "https://api.jaqpot.com"
        configuration = Configuration(
            host=self.host, access_token=access_token if access_token else None
        )
        super().__init__(configuration=configuration)

    def set_access_token(self, access_token):
        """
        Set the access token for the API client.

        :param access_token: str
            The new access token.
        """
        self.configuration.access_token = access_token
