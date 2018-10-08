import jwt


def decode_jwt(jwtoken):
    return jwt.decode(jwtoken, verify=False)
