# from tornado import gen, httpclient
# from tornado.httputil import HTTPHeaders
from jaqpotpy.mappers import decode
import requests

model_path = "model"


def post_pretrained_model(baseurl, api_key, json_request):
    uri = baseurl + model_path
    h = {'Content-Type': 'application/json',
         'Accept': 'application/json',
         'Authorization': "Bearer " + api_key}
    try:
        r = requests.post(uri, headers=h, data=json_request, verify=False)
        return r.json()
    except Exception as e:
        print("Error http: " + str(e))

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
