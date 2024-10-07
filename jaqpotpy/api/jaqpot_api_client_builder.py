from jaqpotpy.api.jaqpot_api_http_client import JaqpotApiHttpClient


class JaqpotApiHttpClientBuilder:
    def __init__(self, host):
        self.http_client = JaqpotApiHttpClient(host=host)  
        
    def build_with_access_token(self, access_token):
        self.http_client.set_access_token(access_token)
        return self
    
    def build_with_api_keys(self, client_key, client_secret):
        self.http_client.set_default_header('X-Api-Key', client_key)
        self.http_client.set_default_header('X-Api-Secret', client_secret)
        return self
        
    def build(self):
        return self.http_client
        
