from jaqpotpy import Jaqpot


class JaqpotLocalhost(Jaqpot):
    def __init__(self):
        super().__init__(
            base_url="http://localhost.jaqpot.org",
            app_url="http://localhost.jaqpot.org:3000",
            login_url="http://localhost.jaqpot.org:8070",
            api_url="http://localhost.jaqpot.org:8080",
            keycloak_realm="jaqpot-local",
            keycloak_client_id="jaqpot-local-test",
        )
