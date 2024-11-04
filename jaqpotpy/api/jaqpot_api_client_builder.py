from jaqpotpy.api.jaqpot_api_http_client import JaqpotApiHttpClient


class JaqpotApiHttpClientBuilder:
    """
    Builder class for creating a JaqpotApiHttpClient instance.
    """

    def __init__(self, host):
        """
        Initialize the builder with the host.

        Args:
            host (str): The host URL for the Jaqpot API.
        """
        self.http_client = JaqpotApiHttpClient(host=host)

    def build_with_access_token(self, access_token):
        """
        Set the access token for the HTTP client.

        Args:
            access_token (str): The access token for authentication.

        Returns:
            JaqpotApiHttpClientBuilder: The builder instance.
        """
        self.http_client.set_access_token(access_token)
        return self

    def build_with_api_keys(self, client_key, client_secret):
        """
        Set the API keys for the HTTP client.

        Args:
            client_key (str): The client key for authentication.
            client_secret (str): The client secret for authentication.

        Returns:
            JaqpotApiHttpClientBuilder: The builder instance.
        """
        self.http_client.set_default_header("X-Api-Key", client_key)
        self.http_client.set_default_header("X-Api-Secret", client_secret)
        return self

    def build(self):
        """
        Build and return the JaqpotApiHttpClient instance.

        Returns:
            JaqpotApiHttpClient: The configured HTTP client instance.
        """
        return self.http_client
