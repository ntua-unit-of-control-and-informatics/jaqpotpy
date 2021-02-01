
import requests
import urllib.parse


doa_path = "doa"


def post_models_doa(baseurl, api_key, json_request, logger):
    uri = baseurl + doa_path
    h = {'Content-Type': 'application/json',
         'Accept': 'application/json',
         'Authorization': "Bearer " + api_key}
    try:
        r = requests.post(uri, headers=h, data=json_request)
        return r.status_code
    except Exception as e:
        logger.error("Error http: " + str(e))


def get_models_doa(baseurl, api_key, modelId, logger):
    uri = baseurl + doa_path
    h = {'Content-Type': 'application/json',
         'Accept': 'application/json',
         'Authorization': "Bearer " + api_key}
    d = {"hasSources":modelId}
    try:
        r = requests.post(uri, headers=h, data=d)
        return r.status_code
    except Exception as e:
        logger.error("Error http: " + str(e))
