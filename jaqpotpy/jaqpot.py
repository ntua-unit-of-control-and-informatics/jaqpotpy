import jaqpotpy.login as jaqlogin


class Jaqpot():

    http_client = {}
    base_url = ''

    def __init__(self, http_client, base_url):
        self.http_client = http_client
        self.base_url = base_url

    def login(self, username, password):
        jaqlogin.authenticate_synch(self.base_url, username=username, password=password)