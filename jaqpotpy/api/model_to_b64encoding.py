from compress_pickle import dumps
from base64 import b64encode


def model_to_b64encoding(actualModel):
    p_mod = dumps(actualModel, compression="bz2")
    raw = b64encode(p_mod)
    model_b64encoded = raw.decode("utf-8")
    return model_b64encoded
