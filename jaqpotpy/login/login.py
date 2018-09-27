from tornado import gen, httpclient
from tornado.httpclient import HTTPClient, AsyncHTTPClient
import getpass


login_path = "aa/login"


def authenticate_synch(baseurl, username, password):
    uri = baseurl + login_path
    httpclient = HTTPClient()
    headers = {'Content-Type': 'multipart/form-data'}
    multipart_form_data = {
        'username': username,
        'password': password
    }
    response = httpclient.fetch(uri, method='POST', headers=headers, body=multipart_form_data)
    return response.body


@gen.coroutine
def authenticate_async(baseurl, username, password):
    uri = baseurl + login_path
    httpclient = AsyncHTTPClient()
    headers = {'Content-Type': 'multipart/form-data'}
    multipart_form_data = {
        'username': username,
        'password': password
    }
    response = yield httpclient.fetch(uri, method='POST', headers=headers)
    raise gen.Return(response.body)


@gen.coroutine
def authenticate_async_safe(baseurl):
    uri = baseurl + login_path
    httpclient = AsyncHTTPClient()
    username = input("Please give username")
    password = getpass("")
    headers = {'Content-Type': 'multipart/form-data'}
    multipart_form_data = {
        'username': username,
        'password': password
    }
    response = yield httpclient.fetch(uri, method='POST', headers=headers, body=multipart_form_data)
    raise gen.Return(response.body)


def authenticate_synch_safe(baseurl, username, password):
    uri = baseurl + login_path
    httpclient = HTTPClient()
    username = input("Please give username")
    password = getpass("")
    headers = {'Content-Type': 'multipart/form-data'}
    multipart_form_data = {
        'username': username,
        'password': password
    }
    response = httpclient.fetch(uri, method='POST', headers=headers)
    return response.body
