import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(BACKEND_ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


# The deployed model emits three classes in this order. Index matters — it maps
# directly onto the model's output vector.
CLASS_LABELS = ["Oral Cancer", "No Abnormality detected", "Oral premalignant lesion"]

# The raw_data folders only cover two of those classes. OPMD (index 2) has no
# local images, so it can be predicted but never appears as ground truth.
FOLDER_TO_CLASS = {"CANCER": 0, "NON CANCER": 1}


@dataclass(frozen=True)
class Settings:
    """Single source of truth for paths and tunables, overridable via .env."""

    app_name: str = _env("APP_NAME", "Prayaas API")
    environment: str = _env("ENVIRONMENT", "development")
    host: str = _env("HOST", "0.0.0.0")
    port: int = int(_env("PORT", "8000"))
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip() for o in _env("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()
        ]
    )

    artifacts_dir: Path = BACKEND_ROOT / "artifacts"
    raw_data_dir: Path = BACKEND_ROOT / "artifacts" / "raw_data"
    models_dir: Path = BACKEND_ROOT / "artifacts" / "models"

    # Pretrained weights published alongside the original Prayaas deployment.
    model_url: str = _env(
        "MODEL_URL", "https://huggingface.co/akhilarayampalli/Prayaas/resolve/main/model_weights.keras"
    )
    model_json_url: str = _env(
        "MODEL_JSON_URL", "https://huggingface.co/akhilarayampalli/Prayaas/resolve/main/model.json"
    )

    # Neon PostgreSQL. Never hardcode this — it carries the password.
    database_url: str = _env("DATABASE_URL", "")
    secret_key: str = _env("SECRET_KEY", "")
    jwt_expire_minutes: int = int(_env("JWT_EXPIRE_MINUTES", "10080"))  # 7 days

    image_size: int = int(_env("IMAGE_SIZE", "224"))
    apply_clahe: bool = _env("APPLY_CLAHE", "true").lower() == "true"
    clahe_clip_limit: float = float(_env("CLAHE_CLIP_LIMIT", "2.0"))
    batch_size: int = int(_env("BATCH_SIZE", "32"))

    # Whether a predicted OPMD counts as CANCER when scoring against the two
    # raw_data folders. Defaults true on evidence: of the 500 CANCER images 75
    # are predicted OPMD, versus 2 of the 200 NON CANCER ones — so OPMD cases
    # were filed under CANCER. It is also the safer screening default, since a
    # premalignant lesion warrants referral rather than an all-clear.
    # Measured over all 700 images: true -> 89.0% accuracy (CANCER recall 0.86),
    # false -> 78.6% (recall 0.71).
    opmd_is_cancer: bool = _env("OPMD_IS_CANCER", "true").lower() == "true"

    @property
    def model_path(self) -> Path:
        return self.models_dir / "model_weights.keras"

    @property
    def model_json_path(self) -> Path:
        return self.models_dir / "model.json"

    @property
    def unet_model_path(self) -> Path:
        """Weights for the lesion-segmentation model trained in
        src/prayaas/research/segmentation.ipynb. Weights-only (not a full
        .keras model) so loading doesn't depend on tf_keras's full-model
        deserialization, which breaks across minor version differences --
        the architecture lives in pipeline/unet_architecture.py instead. No
        MODEL_URL fallback for this one -- if it's missing, segmentation is
        simply unavailable."""
        return self.models_dir / "unet_segmentation.weights.h5"

    @property
    def metrics_path(self) -> Path:
        return self.artifacts_dir / "metrics.json"

    @property
    def image_index_path(self) -> Path:
        return self.artifacts_dir / "image_index.csv"


settings = Settings()
