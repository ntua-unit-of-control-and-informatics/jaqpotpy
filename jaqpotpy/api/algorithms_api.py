#from tornado import gen, httpclient
#from tornado.httputil import HTTPHeaders
#import getpass
#import urllib.parse
from jaqpotpy.mappers import decode
import requests

algos_path = "algorithm"


def get_allgorithms_sync(baseurl, api_key, start=None, max=None):
    uri = baseurl + algos_path
    h = {'Content-Type': 'application/x-www-form-urlencoded',
         'Accept': 'application/json',
         'Authorization': "Bearer " + api_key}
    r = requests.get(uri, headers=h, verify=False)
    return r.json()


def get_allgorithms_classes(base_url, api_key, start, max):
    uri = base_url + algos_path
    h = {'Content-Type': 'application/x-www-form-urlencoded',
         'Accept': 'application/json',
         'Authorization': "Bearer " + api_key}
    r = requests.get(uri, headers=h, verify=False)
    algos = decode.decode_algorithms_to_class(r.json())
    return algos



# def get_allgorithms_sync(baseurl, api_key, start=None, max=None):
#     uri = baseurl + algos_path
#     jclient = httpclient.HTTPClient()
#     h = HTTPHeaders({'Content-Type': 'application/x-www-form-urlencoded'})
#     h.add('Accept', 'application/json')
#     h.add('Authorization', "Bearer " + api_key)
#     uri += '?'
#     if start is not None:
#         uri += "start="+start
#     if max is not None:
#         uri += "max="+max
#     else:
#         uri += "max="+"10"
#     try:
#         response = jclient .fetch(uri, method='GET', headers=h, validate_cert=False)
#         au_req = decode.decode_algorithms_simple(response.body)
#         return au_req
#     except httpclient.HTTPError as e:
#         print("Error: " + str(e))
#     except Exception as e:
#         print("Error: " + str(e))
#     jclient.close()
#
#
# def get_allgorithms_classes(baseurl, api_key, start=None, max=None):
#     uri = baseurl + algos_path
#     jclient = httpclient.HTTPClient()
#     h = HTTPHeaders({'Content-Type': 'application/x-www-form-urlencoded'})
#     h.add('Accept', 'application/json')
#     h.add('Authorization', "Bearer " + api_key)
#     uri += '?'
#     if start is not None:
#         uri += "start="+start
#     if max is not None:
#         uri += "max="+max
#     else:
#         uri += "max="+"10"
#     try:
#         response = jclient.fetch(uri, method='GET', headers=h, validate_cert=False)
#         au_req = decode.decode_algorithms_to_class(response.body)
#         return au_req
#     except httpclient.HTTPError as e:
#         print("Error: " + str(e))
#     except Exception as e:
#         print("Error: " + str(e))
#     jclient.close()
