"""Neon PostgreSQL connection pool and schema bootstrap."""

from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from prayaas.config.configuration import settings
from prayaas.logger import logger

# Rows come back as dicts, not tuples. Spelling that out in the type keeps
# row["email"] checkable rather than silently Any.
DictPool = ConnectionPool[Connection[DictRow]]

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    full_name       TEXT        NOT NULL,
    email           TEXT        NOT NULL UNIQUE,
    password_hash   TEXT        NOT NULL,
    is_profile_complete BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Emails are compared lower-cased, so the uniqueness guarantee must be too.
CREATE UNIQUE INDEX IF NOT EXISTS users_email_lower_idx ON users (LOWER(email));

CREATE TABLE IF NOT EXISTS screenings (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    patient_name    TEXT,
    gender          TEXT,
    age             INTEGER,
    abha            TEXT,
    predicted_class SMALLINT    NOT NULL,
    label           TEXT        NOT NULL,
    confidence      REAL        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- The dashboard always reads one user's rows newest-first.
CREATE INDEX IF NOT EXISTS screenings_user_created_idx ON screenings (user_id, created_at DESC);
"""

_pool: DictPool | None = None


def get_pool() -> DictPool:
    """Lazily opens the pool so importing this module never hits the network."""
    global _pool

    if _pool is None:
        if not settings.database_url:
            raise RuntimeError(
                "DATABASE_URL is not set. Add your Neon connection string to backend/.env"
            )
        # Neon's pooler drops idle connections; check the connection before handing
        # it out rather than surfacing a stale-socket error to the caller.
        _pool = DictPool(
            settings.database_url,
            connection_class=Connection[DictRow],
            min_size=1,
            max_size=5,
            open=True,
            check=DictPool.check_connection,
            kwargs={"row_factory": dict_row},
        )
        logger.info("Postgres pool opened")

    return _pool


def init_db() -> None:
    with get_pool().connection() as conn:
        conn.execute(SCHEMA)
    logger.info("Database schema ready")


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


if __name__ == "__main__":
    init_db()
    with get_pool().connection() as conn:
        version = conn.execute("SELECT version()").fetchone()
        count = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
    print(version["version"] if version else "?")
    print("users:", count["n"] if count else "?")
