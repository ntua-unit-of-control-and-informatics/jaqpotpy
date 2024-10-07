from base64 import b64encode


def model_to_b64encoding(raw_model):
    raw = b64encode(raw_model)
    model_b64encoded = raw.decode("utf-8")
    return model_b64encoded


def file_to_b64encoding(filePath):
    with open(filePath, "rb") as file:
        file_data = file.read()
        b64_encoded_data = b64encode(file_data)
        return b64_encoded_data.decode("utf-8")
