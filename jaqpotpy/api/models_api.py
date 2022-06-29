# from tornado import gen, httpclient
# from tornado.httputil import HTTPHeaders
from jaqpotpy.mappers import decode
import requests
import urllib.parse


model_path = "model"


def post_model_part(baseurl, api_key, modelid, json_request, logger):
    uri = baseurl + model_path + "/" + modelid + "/" + "part"
    h = {'Content-Type': 'application/json',
         'Accept': 'application/json',
         'Authorization': "Bearer " + api_key}
    try:
        r = requests.post(uri, headers=h, data=json_request)
        if r.status_code < 300:
            return r
        else:
            logger.error(r.text)
    except Exception as e:
        logger.error("Error http: " + str(e))


def post_pretrained_model(baseurl, api_key, json_request, logger):
    uri = baseurl + model_path
    h = {'Content-Type': 'application/json',
         'Accept': 'application/json',
         'Authorization': "Bearer " + api_key}
    try:
        r = requests.post(uri, headers=h, data=json_request)
        if r.status_code < 300:
            return r
        else:
            logger.error(r.text)
    except Exception as e:
        logger.error("Error http: " + str(e))


def get_model(baseurl, api_key, modelid, logger):
    uri = baseurl + model_path + "/" + modelid
    h = {'Content-Type': 'application/json',
         'Accept': 'application/json',
         'Authorization': "Bearer " + api_key}
    try:
        r = requests.get(uri, headers=h)
        if r.status_code < 300:
            return r.json()
        else:
            logger.error(r.text)
    except Exception as e:
        logger.error("Error http: " + str(e))


def get_raw_model(baseurl, api_key, modelid, logger):
    uri = baseurl + model_path + "/" + modelid + "/raw"
    h = {'Content-Type': 'application/json',
         'Accept': 'application/json',
         'Authorization': "Bearer " + api_key}
    try:
        r = requests.get(uri, headers=h)
        if r.status_code < 300:
            return r.json()
        else:
            logger.error(r.text)
    except Exception as e:
        logger.error("Error http: " + str(e))


def get_my_models(baseurl, api_key, minimum, maximum, logger):
    uri = baseurl + model_path
    h = {'Content-Type': 'application/json',
         'Accept': 'application/json',
         'Authorization': "Bearer " + api_key}

    d = {'min' : minimum,
         'max' : maximum}
    try:
        r = requests.get(uri, headers=h, params=d)
        if r.status_code < 200:
            retJson = {}
            retJson["total"] = int(r.headers["total"])
            r = r.json()
            # print(r.headers)
            retJson["models"] = r
            return retJson
        else:
            logger.error(r.text)
    except Exception as e:
        logger.error("Error http: " + str(e))
    else:
        retJson = {}
        retJson["total"] = int(r.headers["total"])
        r = r.json()
        # print(r.headers)
        retJson["models"] = r

    return retJson


def get_orgs_models(baseurl, api_key, orgId, minimum, maximum, logger):
    uri = baseurl + model_path
    h = {'Content-Type': 'application/json',
         'Accept': 'application/json',
         'Authorization': "Bearer " + api_key}

    d = {'organization' : orgId,
         'min' : minimum,
         'max' : maximum}
    try:
        r = requests.get(uri, headers=h, params=d)
        if r.status_code < 300:
            retJson = {}
            r = r.json()
            retJson["total"] = int(r.headers["total"])
            retJson["models"] = r
            return retJson
        else:
            logger.error(r.text)
    except Exception as e:
        logger.error("Error http: " + str(e))
    else:
        retJson = {}
        r = r.json()
        retJson["total"] = int(r.headers["total"])
        retJson["models"] = r
    
    return retJson

def get_models_by_tag(baseurl, api_key, tag, minimum, maximum, logger):
    uri = baseurl + model_path
    h = {'Content-Type': 'application/json',
         'Accept': 'application/json',
         'Authorization': "Bearer " + api_key}

    d = {'tag' : tag,
         'min' : minimum,
         'max' : maximum}
    try:
        r = requests.get(uri, headers=h, params=d)
        if r.status_code < 300:
            retJson = {}
            r = r.json()
            retJson["total"] = int(r.headers["total"])
            retJson["models"] = r
            return retJson
        else:
            logger.error(r.text)
    except Exception as e:
        logger.error("Error http: " + str(e))
    else:
        retJson = {}
        r = r.json()
        retJson["total"] = int(r.headers["total"])
        retJson["models"] = r
    
    return retJson

def get_models_by_tag_and_org(baseurl, api_key, organization, tag, minimum, maximum, logger):
    uri = baseurl + model_path
    h = {'Content-Type': 'application/json',
         'Accept': 'application/json',
         'Authorization': "Bearer " + api_key}

    d = {'organization' : organization,
         'tag' : tag,
         'min' : minimum,
         'max' : maximum}
    try:
        r = requests.get(uri, headers=h, params=d)
        if r.status_code < 300:
            retJson = {}
            r = r.json()
            retJson["total"] = int(r.headers["total"])
            retJson["models"] = r
            return retJson
        else:
            logger.error(r.text)
    except Exception as e:
        logger.error("Error http: " + str(e))
    else:
        retJson = {}
        r = r.json()
        retJson["total"] = int(r.headers["total"])
        retJson["models"] = r
    
    return retJson


def predict(baseurl, api_key, modelid, dataseturi, logger):
    uri = baseurl + model_path + "/" + modelid
    h = {"Content-type": "application/x-www-form-urlencoded",
         "Accept": "application/json",
         'Authorization': "Bearer " + api_key}
    data = {
        'dataset_uri': dataseturi
    }
    body = urllib.parse.urlencode(data)
    try:
        r = requests.post(uri, data=body, headers=h)
        if r.status_code < 300:
            return r.json()
        else:
            logger.error("Error with code " + str(r.status_code))
            logger.error(r.json())
    except Exception as e:
        logger.error("Error http: " + str(e))


# def post_pretrained_model(baseurl, api_key, json_request):
#     uri = baseurl + model_path
#     jclient = httpclient.HTTPClient()
#     h = HTTPHeaders({'Content-Type': 'application/json'})
#     h.add('Accept', 'application/json')
#     h.add('Authorization', "Bearer " + api_key)
#     try:
#         response = jclient.fetch(uri, method='POST', headers=h, body=json_request, validate_cert=False)
#         model = decode.decode_model(response.body)
#         return model
#     except httpclient.HTTPError as e:
#         print("Error http: " + str(e))
#     except Exception as e:
#         print("Error generic: " + str(e))
#     jclient.close()
