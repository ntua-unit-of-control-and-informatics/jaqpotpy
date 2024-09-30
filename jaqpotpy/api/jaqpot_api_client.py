from jaqpotpy.api.openapi import Configuration, ApiClient


class JaqpotApiClient(ApiClient):
    def __init__(self, host=None, api_key=None):
        self.host = host or 'https://api.jaqpot.com'
        configuration = Configuration(
            host=self.host,
            api_key={'Bearer': api_key} if api_key else None
        )
        super().__init__(configuration=configuration)

    def set_api_key(self, api_key):
        self.configuration.api_key = {'Bearer': api_key}

    def get_host(self):
        return self.host
