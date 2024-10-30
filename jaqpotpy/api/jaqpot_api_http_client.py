from jaqpotpy.api.openapi import Configuration, ApiClient


class JaqpotApiHttpClient(ApiClient):
    def __init__(self, host=None, access_token=None):
        self.host = host or "https://api.jaqpot.com"
        configuration = Configuration(
            host=self.host, access_token=access_token if access_token else None
        )
        super().__init__(configuration=configuration)

    def set_access_token(self, access_token):
        self.configuration.access_token = access_token
