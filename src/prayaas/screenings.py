"""Screening history: one row per prediction, plus the dashboard's aggregates."""

from psycopg.rows import DictRow

from prayaas.db import get_pool

# Class 1 is "No Abnormality detected"; everything else warrants a referral.
NORMAL_CLASS = 1


def record(
    user_id: int,
    predicted_class: int,
    label: str,
    confidence: float,
    patient_name: str | None = None,
    gender: str | None = None,
    age: int | None = None,
    abha: str | None = None,
) -> DictRow:
    with get_pool().connection() as conn:
        row = conn.execute(
            """
            INSERT INTO screenings
                (user_id, patient_name, gender, age, abha, predicted_class, label, confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, created_at
            """,
            (user_id, patient_name, gender, age, abha, predicted_class, label, confidence),
        ).fetchone()

    assert row is not None
    return row


HISTORY_SQL = """
    SELECT id, patient_name, gender, age, abha, predicted_class, label, confidence, created_at
    FROM screenings
    WHERE user_id = %s
    ORDER BY created_at DESC
    LIMIT %s
"""

STATS_SQL = """
    SELECT
        COUNT(*)                                        AS total,
        COUNT(*) FILTER (WHERE predicted_class <> %s)   AS flagged,
        AVG(confidence)                                 AS avg_confidence,
        MAX(created_at)                                 AS last_at
    FROM screenings
    WHERE user_id = %s
"""


def _shape_stats(row: DictRow | None) -> dict:
    """COUNT/AVG over zero rows give 0/NULL, so a new account is not an error."""
    if row is None:
        return {"total": 0, "flagged": 0, "avg_confidence": None, "last_at": None}

    avg = row["avg_confidence"]
    return {
        "total": int(row["total"]),
        "flagged": int(row["flagged"]),
        "avg_confidence": round(float(avg), 4) if avg is not None else None,
        "last_at": row["last_at"].isoformat() if row["last_at"] else None,
    }


def dashboard(user_id: int, limit: int = 10) -> tuple[list[DictRow], dict]:
    """History and totals over a single connection.

    Both queries share one pooled connection on purpose: every acquisition costs
    a liveness check plus the query, and against Neon in another region that
    round trip dominates. Fetching these separately made the dashboard ~1s
    slower for no benefit.
    """
    with get_pool().connection() as conn:
        rows = conn.execute(HISTORY_SQL, (user_id, limit)).fetchall()
        stats_row = conn.execute(STATS_SQL, (NORMAL_CLASS, user_id)).fetchone()

    return rows, _shape_stats(stats_row)


def history(user_id: int, limit: int = 10) -> list[DictRow]:
    with get_pool().connection() as conn:
        return conn.execute(HISTORY_SQL, (user_id, limit)).fetchall()


def stats(user_id: int) -> dict:
    with get_pool().connection() as conn:
        return _shape_stats(conn.execute(STATS_SQL, (NORMAL_CLASS, user_id)).fetchone())
