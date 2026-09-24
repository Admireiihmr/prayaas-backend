"""One-off copy of accounts and screening history from Neon PostgreSQL into Firestore.

    python scripts/migrate_neon_to_firestore.py --dry-run     # counts only
    python scripts/migrate_neon_to_firestore.py               # copy

Reads DATABASE_URL (the old Neon connection string) and the Firebase settings
from backend/.env. Needs `pip install "psycopg[binary]"`, which is no longer in
requirements.txt since the app itself doesn't use Postgres anymore.

Safe to re-run: documents get deterministic IDs, so a second run overwrites
rather than duplicates. Neon is only ever read. Password hashes are copied as-is
(bcrypt is portable), so existing passwords keep working -- but user IDs change
(serial ints -> email hashes), so tokens issued before the switch stop working
and people have to log in again.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]

import psycopg
from psycopg.rows import dict_row

from prayaas import auth
from prayaas.db import get_db, init_db

BATCH_LIMIT = 400  # Firestore caps a batch at 500 writes


def _commit_in_batches(db, writes):
    batch, pending = db.batch(), 0
    for ref, data in writes:
        batch.set(ref, data)
        pending += 1
        if pending == BATCH_LIMIT:
            batch.commit()
            batch, pending = db.batch(), 0
    if pending:
        batch.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="count rows, write nothing")
    args = parser.parse_args()

    url = os.getenv("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL is not set in backend/.env -- nothing to migrate from.")

    with psycopg.connect(url, row_factory=dict_row) as conn:
        users = conn.execute(
            "SELECT id, full_name, email, password_hash, is_profile_complete, created_at FROM users"
        ).fetchall()
        screenings = conn.execute(
            "SELECT id, user_id, patient_name, gender, age, abha, predicted_class, label, confidence, created_at "
            "FROM screenings"
        ).fetchall()

    print(f"Neon: {len(users)} users, {len(screenings)} screenings")
    if args.dry_run:
        return

    init_db()
    db = get_db()

    new_id = {u["id"]: auth.user_id_for(u["email"]) for u in users}

    user_writes = [
        (
            db.collection("users").document(new_id[u["id"]]),
            {
                "full_name": u["full_name"],
                "email": u["email"].strip().lower(),
                "password_hash": u["password_hash"],
                "is_profile_complete": u["is_profile_complete"],
                "created_at": u["created_at"],
            },
        )
        for u in users
    ]

    orphaned = 0
    screening_writes = []
    for s in screenings:
        if s["user_id"] not in new_id:
            orphaned += 1
            continue
        ref = db.collection("users").document(new_id[s["user_id"]]).collection("screenings").document(f"pg-{s['id']}")
        screening_writes.append(
            (
                ref,
                {
                    "patient_name": s["patient_name"],
                    "gender": s["gender"],
                    "age": s["age"],
                    "abha": s["abha"],
                    "predicted_class": s["predicted_class"],
                    "label": s["label"],
                    "confidence": s["confidence"],
                    "created_at": s["created_at"],
                },
            )
        )

    _commit_in_batches(db, user_writes)
    _commit_in_batches(db, screening_writes)

    # Read back, rather than trusting the writes.
    found_users = db.collection("users").count().get()[0][0].value
    found_screenings = sum(
        db.collection("users").document(uid).collection("screenings").count().get()[0][0].value
        for uid in new_id.values()
    )
    print(f"Firestore: {found_users} users, {found_screenings} screenings")
    if orphaned:
        print(f"Skipped {orphaned} screenings whose user no longer exists.")
    if found_screenings != len(screening_writes):
        sys.exit("Count mismatch after copy -- investigate before switching over.")
    print("Done.")


if __name__ == "__main__":
    main()
