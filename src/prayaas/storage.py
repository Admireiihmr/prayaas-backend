"""Firebase Storage: the images behind each screening.

Objects mirror the Firestore path of the screening they belong to, so one prefix
is one screening and one user's prefix is everything they own:

    users/{uid}/screenings/{screeningId}/original.jpg      the upload, byte for byte
    users/{uid}/screenings/{screeningId}/segmentation.png  U-Net lesion overlay
    users/{uid}/screenings/{screeningId}/gradcam.png       Grad-CAM overlay

The bucket is private. Nothing here hands out public URLs; the screening document
holds the object *paths*, which stay valid (unlike download URLs, which expire).
"""

from firebase_admin import storage
from google.cloud.storage import Bucket

from prayaas.config.configuration import settings
from prayaas.db import get_app


def get_bucket() -> Bucket:
    if not settings.firebase_storage_bucket:
        raise RuntimeError(
            "FIREBASE_STORAGE_BUCKET is not set, e.g. <project-id>.firebasestorage.app "
            "(Firebase console -> Storage)."
        )
    return storage.bucket(app=get_app())


def screening_prefix(user_id: str, screening_id: str) -> str:
    return f"users/{user_id}/screenings/{screening_id}"


def upload(path: str, data: bytes, content_type: str) -> str:
    get_bucket().blob(path).upload_from_string(data, content_type=content_type)
    return path
