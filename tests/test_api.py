from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app
from prayaas.config.configuration import CLASS_LABELS, settings

client = TestClient(app)


def _sample_image() -> Path:
    candidates = sorted((settings.raw_data_dir / "CANCER").glob("*.jpeg"))
    if not candidates:
        pytest.skip("No images in artifacts/raw_data/CANCER")
    return candidates[0]


def test_health_ok():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["classes"] == CLASS_LABELS


def test_predict_returns_three_class_probabilities():
    path = _sample_image()
    with path.open("rb") as f:
        response = client.post("/predict", files={"file": (path.name, f, "image/jpeg")})

    assert response.status_code == 200
    body = response.json()

    assert body["label"] in CLASS_LABELS
    assert set(body["probabilities"]) == set(CLASS_LABELS)
    assert body["probabilities"][body["label"]] == pytest.approx(body["confidence"] * 100, abs=0.05)
    # Probabilities are percentages from a softmax, so they sum to ~100.
    assert sum(body["probabilities"].values()) == pytest.approx(100, abs=0.5)


def test_predict_rejects_non_image():
    response = client.post("/predict", files={"file": ("bad.txt", b"not an image", "text/plain")})
    assert response.status_code == 400


def test_predict_rejects_empty_upload():
    response = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


def test_base64_matches_multipart():
    """The legacy base64 route must agree with the multipart one."""
    import base64

    path = _sample_image()
    with path.open("rb") as f:
        multipart = client.post("/predict", files={"file": (path.name, f, "image/jpeg")}).json()

    encoded = base64.b64encode(path.read_bytes()).decode()
    legacy = client.post("/predict/base64", json={"file": encoded}).json()

    assert multipart["predicted_class"] == legacy["predicted_class"]
