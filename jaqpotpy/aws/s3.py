import requests


def upload_file_to_s3_presigned_url(upload_url: str, raw_bytes: bytes) -> str:
    """
    Uploads the model to S3 using a presigned URL.

    Parameters:
    ----------
    upload_url : str
        The presigned S3 PUT URL returned by Jaqpot's /v1/large-models endpoint.
    file_path : str
        Path to the local ONNX model file.

    Raises:
    -------
    Exception if the upload fails.
    """
    headers = {"Content-Type": "application/octet-stream"}

    response = requests.put(upload_url, data=raw_bytes, headers=headers)

    if response.status_code == 200:
        print("✅ File uploaded successfully.")
    else:
        print(f"❌ Upload failed. Status code: {response.status_code}")
        print(response.text)
        raise Exception("Upload to S3 failed")
