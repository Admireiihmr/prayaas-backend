"""FastAPI application. Run: python main.py serve"""

import base64
import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image
from pydantic import BaseModel, EmailStr, Field

from prayaas import auth, screenings
from prayaas.config.configuration import CLASS_LABELS, settings
from prayaas.db import close_db, init_db
from prayaas.logger import logger
from prayaas.utils.common import load_json
from pipeline.evaluation_pipeline import EvaluationPipeline
from pipeline.prediction_pipeline import GRADCAM_LABEL, SEGMENTATION_LABEL, PredictionPipeline


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        init_db()
    except Exception as e:
        # The screening endpoints work without a database; only auth needs it.
        logger.error("Database unavailable, auth routes will fail: %s", e)
    yield
    close_db()


app = FastAPI(title=settings.app_name, version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = PredictionPipeline()
bearer = HTTPBearer(auto_error=False)

# Routes are declared once, then mounted twice: under /api (what the frontend
# calls through the dev proxy) and at the root (the contract the original
# Streamlit client and Render's health check already use).
router = APIRouter()


# --- models -----------------------------------------------------------------

class SignupRequest(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=72)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=72)


class ImageInput(BaseModel):
    """Base64 payload, kept for compatibility with the original Streamlit client."""

    file: str


# processing_steps label -> name of the stored image.
_STEP_ROLES = {SEGMENTATION_LABEL: "segmentation", GRADCAM_LABEL: "gradcam"}


def _scan_files(raw: bytes, result: dict) -> dict[str, tuple[str, bytes, str]]:
    """The images to keep for a screening: the upload exactly as received, plus the
    explainability overlays (when there are any). The "Original" processing step
    is skipped -- it's just the upload re-encoded, at several times the size."""
    with Image.open(io.BytesIO(raw)) as im:
        fmt = im.format or ""
    ext = "jpg" if fmt == "JPEG" else (fmt.lower() or "img")
    files = {"original": (f"original.{ext}", raw, Image.MIME.get(fmt, "application/octet-stream"))}

    for step in result["processing_steps"]:
        role = _STEP_ROLES.get(step["label"])
        if role:
            png = base64.b64decode(step["image"].split(",", 1)[1])
            files[role] = (f"{role}.png", png, "image/png")
    return files


def _public(user: dict) -> dict:
    return {
        "id": user["id"],
        "name": user["full_name"],
        "full_name": user["full_name"],
        "email": user["email"],
        "is_profile_complete": user["is_profile_complete"],
    }


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    try:
        payload = auth.decode_token(credentials.credentials)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token.") from e

    user = auth.get_user_by_id(payload["sub"])
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists.")
    return user


def optional_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict | None:
    """Identifies the caller when a token is present; no token at all is anonymous.

    Lets /predict stay usable without an account (and by the legacy client, which
    sends no token) while still filing a signed-in user's screening.

    A token that is present but bad is NOT anonymous: it gets the same 401 as any
    other route. Quietly downgrading it meant the prediction ran, nothing was
    saved, and the user was never told -- which is exactly what happened to every
    session issued before the move to Firestore.
    """
    if credentials is None:
        return None
    return current_user(credentials)


# --- auth -------------------------------------------------------------------

@router.post("/signup", status_code=201)
def signup(request: SignupRequest) -> dict:
    try:
        user = auth.create_user(request.full_name, request.email, request.password)
    except auth.EmailTakenError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    logger.info("Signup: %s", user["email"])
    return {"user": _public(user)}


@router.post("/login")
def login(request: LoginRequest) -> dict:
    user = auth.authenticate(request.email, request.password)
    if user is None:
        # Same message for unknown email and wrong password, so the response
        # cannot be used to discover which accounts exist.
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    return {"token": auth.create_token(user["id"], user["email"]), "user": _public(user)}


@router.get("/me")
def me(user: dict = Depends(current_user)) -> dict:
    return {"user": _public(user)}


@router.get("/screenings")
def screening_history(limit: int = 10, user: dict = Depends(current_user)) -> dict:
    """The dashboard's data: totals plus this user's most recent screenings."""
    limit = max(1, min(limit, 50))
    rows, stats = screenings.dashboard(user["id"], limit)

    return {
        "stats": stats,
        "screenings": [
            {
                "id": r["id"],
                "patient_name": r["patient_name"],
                "gender": r["gender"],
                "age": r["age"],
                "label": r["label"],
                "predicted_class": r["predicted_class"],
                "confidence": round(float(r["confidence"]), 4),
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ],
    }


# --- screening --------------------------------------------------------------

@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "environment": settings.environment,
        "model_ready": settings.model_path.exists(),
        "classes": CLASS_LABELS,
    }


@router.get("/metrics")
def metrics() -> dict:
    if not settings.metrics_path.exists():
        raise HTTPException(status_code=404, detail="No metrics yet. Run an evaluation first.")
    return load_json(settings.metrics_path)


@router.post("/predict")
async def predict(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    patient_name: str | None = Form(None),
    gender: str | None = Form(None),
    age: int | None = Form(None),
    abha: str | None = Form(None),
    user: dict | None = Depends(optional_user),
) -> dict:
    """Scores an uploaded image file (multipart/form-data).

    Signed-in callers get the result filed against their dashboard, and (when
    Firebase Storage is configured) their images kept under the screening.
    Anonymous ones still get a prediction, but nothing is stored: there's no
    owner to file a patient's photo under.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        result = pipeline.predict_bytes(raw)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}") from e

    if user is not None:
        try:
            saved = screenings.record(
                user_id=user["id"],
                predicted_class=result["predicted_class"],
                label=result["label"],
                confidence=result["confidence"],
                patient_name=patient_name,
                gender=gender,
                age=age,
                abha=abha,
            )
            result["screening_id"] = saved["id"]
            if settings.firebase_storage_bucket:
                # After the response goes out, so uploading a few MB never delays
                # the result.
                background.add_task(screenings.store_images, user["id"], saved["id"], _scan_files(raw, result))
        except Exception:
            # A history write must never cost the user their result.
            logger.exception("Could not record screening")

    return result


@router.post("/predict/base64")
def predict_base64(payload: ImageInput) -> dict:
    """Legacy contract: {"file": "<base64>"} -> class probabilities."""
    try:
        return pipeline.predict_base64(payload.file)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}") from e


@router.post("/evaluate")
def evaluate(limit: int | None = None) -> dict:
    """Scores the labelled images in artifacts/raw_data. Slow: 700 images by default."""
    try:
        return {"status": "evaluated", "metrics": EvaluationPipeline().run(limit=limit)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


app.include_router(router, prefix="/api")
app.include_router(router)
