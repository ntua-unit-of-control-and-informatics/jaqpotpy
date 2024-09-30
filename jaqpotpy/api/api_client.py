from jaqpotpy.api.openapi import Configuration


class JaqpotApiClient:
    def __init__(self, host='https://api.jaqpot.com'):
        self._configuration = Configuration(host=self.host)
