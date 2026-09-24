"""Firestore client.

Layout:

    users/{sha256(lowercased email)}       full_name, email, password_hash,
                                           is_profile_complete, created_at
    users/{uid}/screenings/{auto id}       patient_name, gender, age, abha,
                                           predicted_class, label, confidence,
                                           created_at

Firestore has no UNIQUE constraint, so the email hash doubles as the user's
document ID and `create()` fails atomically on a duplicate. Screenings live in a
per-user subcollection so "one user's rows, newest first" needs only Firestore's
automatic single-field indexes -- no composite index to create by hand.
"""

import json
import threading
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import Client

from prayaas.config.configuration import BACKEND_ROOT, settings
from prayaas.logger import logger

_app: firebase_admin.App | None = None
_client: Client | None = None
# Reentrant: get_db() holds it while calling get_app(), which takes it too.
_lock = threading.RLock()


def _load_credentials() -> credentials.Certificate:
    if settings.firebase_credentials_json:
        return credentials.Certificate(json.loads(settings.firebase_credentials_json))

    if settings.firebase_credentials_path:
        path = Path(settings.firebase_credentials_path)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        return credentials.Certificate(str(path))

    raise RuntimeError(
        "Firebase credentials are not set. Point FIREBASE_CREDENTIALS_PATH at your "
        "service-account JSON in backend/.env, or set FIREBASE_CREDENTIALS_JSON."
    )


def get_app() -> firebase_admin.App:
    """The one Firebase app, shared by Firestore and Storage.

    Lazy so importing this module never hits the network, and locked because sync
    endpoints run in a threadpool: two first requests racing into initialize_app
    would make the second raise "default app already exists".
    """
    global _app

    if _app is None:
        with _lock:
            if _app is None:
                options = {"storageBucket": settings.firebase_storage_bucket} if settings.firebase_storage_bucket else None
                _app = firebase_admin.initialize_app(_load_credentials(), options)

    return _app


def get_db() -> Client:
    global _client

    if _client is None:
        with _lock:
            if _client is None:
                _client = firestore.client(get_app())
                logger.info("Firestore client ready (project %s)", _client.project)

    return _client


def init_db() -> None:
    """Fails fast if the credentials are wrong or Firestore isn't enabled for the
    project -- a cheap read, since there's no schema to create."""
    list(get_db().collection("users").limit(1).stream())
    logger.info("Firestore reachable")


def close_db() -> None:
    global _app, _client
    with _lock:
        if _app is not None:
            firebase_admin.delete_app(_app)
        _app = _client = None


if __name__ == "__main__":
    init_db()
    users = get_db().collection("users")
    print("project:", get_db().project)
    print("users:", users.count().get()[0][0].value)
