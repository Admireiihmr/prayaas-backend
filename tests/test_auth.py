import pytest

from prayaas import auth


def test_hash_is_not_the_password():
    hashed = auth.hash_password("Str0ng!Pass")
    assert hashed.startswith("$2b$")
    assert "Str0ng!Pass" not in hashed


def test_verify_accepts_correct_and_rejects_wrong():
    hashed = auth.hash_password("Str0ng!Pass")
    assert auth.verify_password("Str0ng!Pass", hashed)
    assert not auth.verify_password("Str0ng!Pas", hashed)
    assert not auth.verify_password("", hashed)


def test_verify_survives_a_malformed_hash():
    """A corrupt row must return False, not raise."""
    assert not auth.verify_password("anything", "not-a-bcrypt-hash")


def test_same_password_gets_a_different_salt():
    assert auth.hash_password("Str0ng!Pass") != auth.hash_password("Str0ng!Pass")


def test_overlong_password_is_rejected_not_truncated():
    """bcrypt ignores bytes past 72; silently accepting them would be a footgun."""
    with pytest.raises(ValueError):
        auth.hash_password("a" * 73)


def test_token_roundtrips():
    token = auth.create_token(42, "someone@example.com")
    payload = auth.decode_token(token)
    assert payload["sub"] == "42"
    assert payload["email"] == "someone@example.com"


def test_token_signed_with_another_key_is_rejected():
    import jwt

    forged = jwt.encode({"sub": "1"}, "not-the-real-secret", algorithm="HS256")
    with pytest.raises(jwt.PyJWTError):
        auth.decode_token(forged)


def test_expired_token_is_rejected(monkeypatch):
    import jwt

    from prayaas.config.configuration import settings

    # Issue a token that expired an hour ago.
    monkeypatch.setattr(auth, "settings", type("S", (), {
        "secret_key": settings.secret_key,
        "jwt_expire_minutes": -60,
    })())

    with pytest.raises(jwt.ExpiredSignatureError):
        auth.decode_token(auth.create_token(1, "a@example.com"))
