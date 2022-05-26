#from tornado import gen, httpclient
#from tornado.httputil import HTTPHeaders
import getpass
import urllib.parse
from jaqpotpy.mappers import decode
import http.client
import requests

login_path = "aa/login"


def authenticate_sync(baseurl, username, password):
    uri = baseurl + login_path
    data = {
        'username': username,
        'password': password
    }
    body = urllib.parse.urlencode(data)
    h = {"Content-type": "application/x-www-form-urlencoded",
         "Accept": "application/json"}
    try:
        r = requests.post(uri, data=body, headers=h)
        # resp = decode.decode_auth(r.text)
        return r.json()
    except Exception as e:
        print("Error 1: " + str(e))


def validate_api_key(baseurl, api_key):
    uri = baseurl + "aa/validate/accesstoken"
    data = api_key
    h = {"Content-type": "*/*", "Accept": "application/json"}
    try:
        r = requests.post(uri, data=data, headers=h)
        # resp = decode.decode_auth(r.text)
        return r.json()
    except Exception as e:
        print("Error 1: " + str(e))



# def authenticate_sync(http_client, baseurl, username, password):
#     uri = baseurl + login_path
#     jclient = httpclient.HTTPClient()
#     h = HTTPHeaders({'Content-Type': 'application/x-www-form-urlencoded'})
#     h.add('Accept', 'application/json')
#     data = {
#         'username': username,
#         'password': password
#     }
#     body = urllib.parse.urlencode(data)
#     try:
#         response = jclient.fetch(uri, method='POST', headers=h, body=body, validate_cert=False)
#         au_req = decode.decode_auth(response.body)
#         return au_req
#     except httpclient.HTTPError as e:
#         print("Error: " + str(e))
#     except Exception as e:
#         print("Error: " + str(e))
#     jclient.close()


# @gen.coroutine
# def authenticate_async(baseurl, username, password):
#     uri = baseurl + login_path
#     jclient = httpclient.AsyncHTTPClient()
#     h = HTTPHeaders({'Content-Type': 'application/x-www-form-urlencoded'})
#     h.add('Accept', 'application/json')
#     data = {
#         'username': username,
#         'password': password
#     }
#     body = urllib.parse.urlencode(data)
#     try:
#         response = yield jclient.fetch(uri, method='POST', headers=h, body=body, validate_cert=False)
#         au_req = decode.decode_auth(response.body)
#         raise gen.Return(au_req)
#     except httpclient.HTTPError as e:
#         print("Error: " + str(e))
#     except Exception as e:
#         print("Error: " + str(e))
#     jclient.close()
#
#
# @gen.coroutine
# def authenticate_async_hidepass(baseurl):
#     uri = baseurl + login_path
#     jclient = httpclient.AsyncHTTPClient()
#     username = input("Username: ")
#     password = getpass.getpass("Password: ")
#     h = HTTPHeaders({'Content-Type': 'application/x-www-form-urlencoded'})
#     h.add('Accept', 'application/json')
#     data = {
#         'username': username,
#         'password': password
#     }
#     body = urllib.parse.urlencode(data)
#     try:
#         response = yield jclient.fetch(uri, method='POST', headers=h, body=body, validate_cert=False)
#         au_req = decode.decode_auth(response.body)
#         raise gen.Return(au_req)
#     except httpclient.HTTPError as e:
#         print("Error: " + str(e))
#     except Exception as e:
#         print("Error: " + str(e))
#     jclient.close()
#
#
# def authenticate_sync_hidepass(baseurl):
#     uri = baseurl + login_path
#     jclient = httpclient.HTTPClient()
#     username = input("Username: ")
#     password = getpass.getpass("Password: ")
#     h = HTTPHeaders({'Content-Type': 'application/x-www-form-urlencoded'})
#     h.add('Accept', 'application/json')
#     data = {
#         'username': username,
#         'password': password
#     }
#     body = urllib.parse.urlencode(data)
#     try:
#         response = jclient.fetch(uri, method='POST', headers=h, body=body, validate_cert=False)
#         au_req = decode.decode_auth(response.body)
#         return au_req
#     except httpclient.HTTPError as e:
#         print("Error: " + str(e))
#     except Exception as e:
#         print("Error: " + str(e))
#     jclient.close()
