from base64 import b64encode


def model_to_b64encoding(raw_model):
    """
    Encodes a raw model to a base64 string.

    Args:
        raw_model (bytes): The raw model data in bytes.

    Returns:
        str: The base64 encoded string of the model.
    """
    raw = b64encode(raw_model)
    model_b64encoded = raw.decode("utf-8")
    return model_b64encoded


def file_to_b64encoding(filePath):
    """
    Reads a file and encodes its contents to a base64 string.

    Args:
        filePath (str): The path to the file to be encoded.

    Returns:
        str: The base64 encoded string of the file contents.
    """
    with open(filePath, "rb") as file:
        file_data = file.read()
        b64_encoded_data = b64encode(file_data)
        return b64_encoded_data.decode("utf-8")
