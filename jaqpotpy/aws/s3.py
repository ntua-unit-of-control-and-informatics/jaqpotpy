import requests
from tqdm import tqdm
from io import BytesIO


def upload_file_to_s3_presigned_url(upload_url: str, raw_bytes: bytes) -> str:
    """
    Uploads a file to S3 using a presigned URL with a progress bar.

    Parameters:
    ----------
    upload_url : str
        The presigned S3 PUT URL returned by Jaqpot's /v1/large-models endpoint.
    raw_bytes : bytes
        The raw bytes of the model to upload.

    Raises:
    -------
    Exception if the upload fails.
    """
    headers = {"Content-Type": "application/octet-stream"}

    total = len(raw_bytes)
    stream = BytesIO(raw_bytes)

    class ProgressReader:
        def __init__(self, file, total_size):
            self.file = file
            self.progress = tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Uploading"
            )

        def read(self, size=-1):
            chunk = self.file.read(size)
            self.progress.update(len(chunk))
            return chunk

        def __getattr__(self, attr):
            return getattr(self.file, attr)

    response = requests.put(
        upload_url, data=ProgressReader(stream, total), headers=headers
    )

    if response.status_code == 200:
        print("✅ File uploaded successfully.")
    else:
        print(f"❌ Upload failed. Status code: {response.status_code}")
        print(response.text)
        raise Exception("Upload to S3 failed")
