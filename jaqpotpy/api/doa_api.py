import requests
import urllib.parse


doa_path = "doa"


def post_models_doa(baseurl, api_key, json_request, logger):
    uri = baseurl + doa_path
    h = {'Content-Type': 'application/json',
         'Accept': 'application/json',
         'Authorization': "Bearer " + api_key}
    try:
        r = requests.post(uri, headers=h, data=json_request, verify=False)
        return r.status_code
    except Exception as e:
        logger.error("Error http: " + str(e))
