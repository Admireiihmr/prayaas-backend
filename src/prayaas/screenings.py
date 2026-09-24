"""Screening history: one document per prediction, plus the dashboard's aggregates."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from google.cloud.firestore import Query
from google.cloud.firestore_v1.base_query import FieldFilter

from prayaas import storage
from prayaas.db import get_db
from prayaas.logger import logger

# Class 1 is "Normal"; everything else warrants a referral.
NORMAL_CLASS = 1

# The dashboard needs several independent reads; running them concurrently keeps
# its latency at one round trip instead of the sum of all of them.
_executor = ThreadPoolExecutor(max_workers=3)


def _screenings(user_id: str):
    return get_db().collection("users").document(user_id).collection("screenings")


def record(
    user_id: str,
    predicted_class: int,
    label: str,
    confidence: float,
    patient_name: str | None = None,
    gender: str | None = None,
    age: int | None = None,
    abha: str | None = None,
) -> dict:
    created_at = datetime.now(timezone.utc)
    ref = _screenings(user_id).document()
    ref.set(
        {
            "patient_name": patient_name,
            "gender": gender,
            "age": age,
            "abha": abha,
            "predicted_class": predicted_class,
            "label": label,
            "confidence": confidence,
            "created_at": created_at,
        }
    )
    return {"id": ref.id, "created_at": created_at}


def store_images(user_id: str, screening_id: str, files: dict[str, tuple[str, bytes, str]]) -> None:
    """Uploads a screening's images to Storage, then records their paths on it.

    files maps a role ("original", "segmentation", ...) to (filename, bytes,
    content type). Runs after the response has gone out, so it never raises: an
    image that fails to upload is logged and left out of the document's `images`
    map, rather than the screening pointing at an object that isn't there.
    """
    prefix = storage.screening_prefix(user_id, screening_id)
    paths = {}
    for role, (filename, data, content_type) in files.items():
        try:
            paths[role] = storage.upload(f"{prefix}/{filename}", data, content_type)
        except Exception:
            logger.exception("Could not store %s image for screening %s", role, screening_id)

    if not paths:
        return
    try:
        _screenings(user_id).document(screening_id).update({"images": paths})
    except Exception:
        logger.exception("Could not record image paths on screening %s", screening_id)


def _history(user_id: str, limit: int) -> list[dict]:
    query = _screenings(user_id).order_by("created_at", direction=Query.DESCENDING).limit(limit)
    return [{**snap.to_dict(), "id": snap.id} for snap in query.stream()]


def _totals(user_id: str) -> tuple[int, float | None]:
    query = _screenings(user_id).count(alias="total").avg("confidence", alias="avg_confidence")
    results = {r.alias: r.value for r in query.get()[0]}
    total = int(results["total"])
    # Firestore averages zero documents to 0.0, not NULL; the dashboard treats
    # None as "no data yet", and 0% confidence would be a false reading.
    return total, results["avg_confidence"] if total else None


def _flagged(user_id: str) -> int:
    query = _screenings(user_id).where(filter=FieldFilter("predicted_class", "!=", NORMAL_CLASS))
    return int(query.count(alias="flagged").get()[0][0].value)


def _shape_stats(total: int, flagged: int, avg: float | None, last_at: datetime | None) -> dict:
    """COUNT/AVG over zero docs give 0/None, so a new account is not an error."""
    return {
        "total": total,
        "flagged": flagged,
        "avg_confidence": round(float(avg), 4) if avg is not None else None,
        "last_at": last_at.isoformat() if last_at else None,
    }


def dashboard(user_id: str, limit: int = 10) -> tuple[list[dict], dict]:
    """History and totals, fetched concurrently."""
    history = _executor.submit(_history, user_id, limit)
    totals = _executor.submit(_totals, user_id)
    flagged = _executor.submit(_flagged, user_id)

    rows = history.result()
    total, avg = totals.result()
    last_at = rows[0]["created_at"] if rows else None
    return rows, _shape_stats(total, flagged.result(), avg, last_at)


def history(user_id: str, limit: int = 10) -> list[dict]:
    return _history(user_id, limit)


def stats(user_id: str) -> dict:
    latest = _history(user_id, 1)
    total, avg = _totals(user_id)
    return _shape_stats(total, _flagged(user_id), avg, latest[0]["created_at"] if latest else None)
