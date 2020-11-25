from jaqpotpy.mappers import decode
import requests


dataset_path = "dataset"


def create_dataset_sync(baseurl, api_key, json_dataset, logger):
    uri = baseurl + dataset_path
    h = {'Content-Type': 'application/json',
         'Accept': 'application/json',
         'Authorization': "Bearer " + api_key}
    try:
        r = requests.post(uri, headers=h, data=json_dataset)
        if r.status_code < 300:
            return r.json()
        else:
            logger.error("Error with code " + str(r.status_code))
            logger.error(r.json())
    except Exception as e:
        logger.error("Error http: " + str(e))


def get_dataset(baseurl, api_key, datasetid, logger):
    uri = baseurl + dataset_path + "/" + datasetid
    h = {'Content-Type': 'application/json',
         'Accept': 'application/json',
         'Authorization': "Bearer " + api_key}
    params = {
        'dataEntries': "true",
        'rowStart': "0",
        'rowMax': '1000'
    }
    try:
        r = requests.get(uri, headers=h, params=params)
        if r.status_code < 300:
            return r.json()
        else:
            logger.error("Error with code " + str(r.status_code))
            logger.error(r.json())
    except Exception as e:
        logger.error("Error http: " + str(e))


# def create_dataset_sync(baseurl, api_key, json_dataset):
#     uri = baseurl + dataset_path
#     jclient = httpclient.HTTPClient()
#     h = HTTPHeaders({'Content-Type': 'application/json'})
#     h.add('Accept', 'application/json')
#     h.add('Authorization', "Bearer " + api_key)
#     try:
#         response = jclient.fetch(uri, method='POST', headers=h, body=json_dataset, validate_cert=False)
#         au_req = decode.decode_dataset(response.body)
#         return au_req
#     except httpclient.HTTPError as e:
#         print("Error http: " + str(e))
#     except Exception as e:
#         print("Error generic: " + str(e))
#     jclient.close()
