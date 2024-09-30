from jaqpotpy.api.openapi import Configuration, ApiClient


class JaqpotApiClient:
    def __init__(self, host, api_key=None):
        self.host = host or 'https://api.jaqpot.com'
        self._configuration = Configuration(
            host=self.host,
            api_key={'Bearer': api_key} if api_key else None
        )
        self._api = ApiClient(configuration=self._configuration)

    def get_client(self):
        return self._api
 
