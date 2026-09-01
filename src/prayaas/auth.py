"""Password hashing, JWT issuing, and the user table's queries."""

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt
from psycopg import errors
from psycopg.rows import DictRow

from prayaas.config.configuration import settings
from prayaas.db import get_pool

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

def create_token(user_id: int, email: str) -> str:
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

PUBLIC_COLUMNS = "id, full_name, email, is_profile_complete, created_at"


class EmailTakenError(Exception):
    pass


def create_user(full_name: str, email: str, password: str) -> DictRow:
    with get_pool().connection() as conn:
        try:
            row = conn.execute(
                f"""
                INSERT INTO users (full_name, email, password_hash)
                VALUES (%s, %s, %s)
                RETURNING {PUBLIC_COLUMNS}
                """,
                (full_name.strip(), email.strip().lower(), hash_password(password)),
            ).fetchone()
        except errors.UniqueViolation as e:
            raise EmailTakenError("An account with that email already exists.") from e

    assert row is not None  # RETURNING on a successful INSERT always yields a row
    return row


def get_user_by_email(email: str) -> DictRow | None:
    with get_pool().connection() as conn:
        return conn.execute(
            f"SELECT {PUBLIC_COLUMNS}, password_hash FROM users WHERE LOWER(email) = %s",
            (email.strip().lower(),),
        ).fetchone()


def get_user_by_id(user_id: int) -> DictRow | None:
    with get_pool().connection() as conn:
        return conn.execute(
            f"SELECT {PUBLIC_COLUMNS} FROM users WHERE id = %s", (user_id,)
        ).fetchone()


def authenticate(email: str, password: str) -> dict | None:
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        return None
    return {k: v for k, v in user.items() if k != "password_hash"}
