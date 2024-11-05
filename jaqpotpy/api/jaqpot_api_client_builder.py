from jaqpotpy.api.jaqpot_api_http_client import JaqpotApiHttpClient


class JaqpotApiHttpClientBuilder:
    def __init__(self, host):
        self.http_client = JaqpotApiHttpClient(host=host)

    def build_with_access_token(self, access_token):
        self.http_client.set_access_token(access_token)
        return self

    def build_with_api_keys(self, api_key, api_secret):
        if api_key is None or api_secret is None:
            raise ValueError("api_key and api_secret must be set")
        self.http_client.set_default_header("X-Api-Key", api_key)
        self.http_client.set_default_header("X-Api-Secret", api_secret)
        return self

    def build(self):
        return self.http_client
