from base64 import b64encode

def model_to_b64encoding(actualModel):
    raw = b64encode(actualModel)
    model_b64encoded = raw.decode("utf-8")
    return model_b64encoded
