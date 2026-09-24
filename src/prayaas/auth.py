"""Password hashing, JWT issuing, and the users collection's queries."""

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt
from google.api_core.exceptions import AlreadyExists

from prayaas.config.configuration import settings
from prayaas.db import get_db

ALGORITHM = "HS256"


# ── passwords ────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    # bcrypt silently truncates at 72 bytes; reject rather than accept a password
    # whose tail is ignored.
    if len(password.encode()) > 72:
        raise ValueError("Password must be at most 72 bytes.")
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        return False


# ── tokens ───────────────────────────────────────────────────────────────────

def create_token(user_id: str, email: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])


# ── queries ──────────────────────────────────────────────────────────────────

class EmailTakenError(Exception):
    pass


def user_id_for(email: str) -> str:
    """The user's document ID: a hash of the lower-cased email. Emails can contain
    characters Firestore forbids in IDs (e.g. '/'), and making the ID a pure
    function of the email is what lets create() enforce uniqueness atomically."""
    return hashlib.sha256(email.strip().lower().encode()).hexdigest()


def _users():
    return get_db().collection("users")


def _row(user_id: str, data: dict) -> dict:
    return {
        "id": user_id,
        "full_name": data["full_name"],
        "email": data["email"],
        "is_profile_complete": data.get("is_profile_complete", False),
        "created_at": data["created_at"],
    }


def create_user(full_name: str, email: str, password: str) -> dict:
    email = email.strip().lower()
    user_id = user_id_for(email)
    data = {
        "full_name": full_name.strip(),
        "email": email,
        "password_hash": hash_password(password),
        "is_profile_complete": False,
        "created_at": datetime.now(timezone.utc),
    }

    try:
        _users().document(user_id).create(data)
    except AlreadyExists as e:
        raise EmailTakenError("An account with that email already exists.") from e

    return _row(user_id, data)


def get_user_by_email(email: str) -> dict | None:
    snap = _users().document(user_id_for(email)).get()
    if not snap.exists:
        return None

    data = snap.to_dict()
    return {**_row(snap.id, data), "password_hash": data["password_hash"]}


def get_user_by_id(user_id: str) -> dict | None:
    # user_id comes from a signed token, but a '/' would still turn the lookup
    # into a path into a different collection -- reject rather than resolve it.
    if not user_id or "/" in user_id:
        return None

    snap = _users().document(user_id).get()
    return _row(snap.id, snap.to_dict()) if snap.exists else None


def authenticate(email: str, password: str) -> dict | None:
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        return None
    return {k: v for k, v in user.items() if k != "password_hash"}
