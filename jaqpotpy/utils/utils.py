from tornado import gen, httpclient

@gen.coroutine
async def authenticate(http_client, baseurl, username, password):
    try:
        response = await httpclient
