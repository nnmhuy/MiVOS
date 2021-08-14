from google.cloud import storage
import io
import numpy as np
from functools import cache
import cv2


CLOUD_STORAGE_BUCKET = 'smp-development'


def upload_to_google_storage(filename, file, content_type='image/jpeg'):
  if not file:
    return 'No file uploaded.'

  # Create a Cloud Storage client.
  gcs = storage.Client()

  # Get the bucket that the file will be uploaded to.
  bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)

  # Create a new blob and upload the file's content.
  blob = bucket.blob(filename)

  blob.upload_from_string(
      file,
      content_type=content_type
  )
  blob.cache_control = "no-cache"
  blob.patch()

  return blob.public_url


@cache
def download_image_from_google_storage(file_url, readFlag=cv2.IMREAD_COLOR):
    """Downloads a blob from the bucket."""
    pos = file_url.find(CLOUD_STORAGE_BUCKET)
    filename = file_url[pos + len(CLOUD_STORAGE_BUCKET) + 1:]

    storage_client = storage.Client()

    bucket = storage_client.bucket(CLOUD_STORAGE_BUCKET)

    blob = bucket.blob(filename)

    image_np = np.asarray(bytearray(blob.download_as_string()), dtype=np.uint8)
    image = cv2.imdecode(image_np, readFlag)

    return image
