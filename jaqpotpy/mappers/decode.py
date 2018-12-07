from jaqpotpy.dto.auth_request import AuthRequest
# from tornado.escape import json_decode
import json
from jaqpotpy.entities.algorithm import Algorithm
import json


def decode_auth(request):
    jsonreq = json.dumps(request)
    au_req = AuthRequest(jsonreq['userName'], jsonreq['authToken'])
    return au_req


def decode_algorithms_simple(request):
    algorithms = json.dumps(request)
    return algorithms


def decode_algorithms_to_class(request):
    algorithms = request
    # algorithms = json.dumps(request)
    algos = []
    for algo in algorithms:
        al = Algorithm(algo)
        algos.append(al)
    return algos


def decode_feature(request):
    feature = json.dumps(request)
    return feature


def decode_dataset(request):
    dataset = json.dumps(request)
    return dataset


def decode_model(response):
    model = json.dumps(response)
    return model


# def decode_auth(request):
#     jsonreq = json_decode(request)
#     au_req = AuthRequest(jsonreq['userName'], jsonreq['authToken'])
#     return au_req
#
#
# def decode_algorithms_simple(request):
#     algorithms = json_decode(request)
#     return algorithms
#
#
# def decode_algorithms_to_class(request):
#     algorithms = json_decode(request)
#     algos = []
#     for algo in algorithms:
#         al = Algorithm(algo)
#         algos.append(al)
#     return algos
#
#
# def decode_feature(request):
#     feature = json_decode(request)
#     return feature
#
#
# def decode_dataset(request):
#     dataset = json_decode(request)
#     return dataset
#
#
# def decode_model(response):
#     model = json_decode(response)
#     return model
