# from tornado import gen, httpclient
# from tornado.httputil import HTTPHeaders
# from jaqpotpy.mappers import decode
import requests
import json


chempot_path = "chempot"

def predict(baseurl, api_key, modelid, smiles, descriptors, doa, logger):
    
    uri = baseurl + chempot_path + "/" 

    h = {"Content-type": "application/json",
         "Accept": "application/json",
         'Authorization': "Bearer " + api_key}
    
    data = {
      "smiles": smiles,
      "modelId": modelid,
      "withDoa": doa,
      "descriptors": descriptors
    }
    
    try:
        r = requests.post(uri, data=json.dumps(data), headers=h)
        if r.status_code < 300:
            return r.json()
        else:
            logger.error("Error with code " + str(r.status_code))
            logger.error(r.json())
    except Exception as e:
        logger.error("Error http: " + str(e))
