import json
from pathlib import Path
from typing import Any

import joblib


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_object(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_object(path: Path) -> Any:
    return joblib.load(path)
