from base64 import b64encode


def model_to_b64encoding(actualModel):
    raw = b64encode(actualModel)
    model_b64encoded = raw.decode("utf-8")
    return model_b64encoded


def csv_to_b64encoding(actualCsv):
    with open(actualCsv, "rb") as file:
        file_data = file.read()
        b64_encoded_data = b64encode(file_data)
        return b64_encoded_data.decode("utf-8")
