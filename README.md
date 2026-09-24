---
title: Prayaas Backend
emoji: 🩺
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Prayaas Backend

FastAPI service that screens oral cavity images with a pretrained 3-class classifier.

## Layout

```
backend/
├── app.py                  FastAPI: auth + screening routes
├── main.py                 CLI: evaluate | predict | serve
├── requirements.txt
├── .env                    local config (gitignored; copy from .env.example)
├── render.yaml             Render deployment
├── runtime.txt             python-3.11.9
├── packages.txt            libgl1 — the system lib OpenCV needs
├── .gitattributes          git-lfs for *.keras
├── artifacts/
│   ├── models/             model_weights.keras (73 MB), model.json
│   └── raw_data/
│       ├── CANCER/         500 images
│       └── NON CANCER/     200 images
├── pipeline/
│   ├── data_ingestion.py       indexes the class folders
│   ├── data_preprocessing.py   resize 224 -> CLAHE -> scale to [0,1]
│   ├── model_loader.py         loads the Keras model (see the tf_keras note)
│   ├── model_evaluation.py     scores predictions, writes metrics.json
│   ├── evaluation_pipeline.py  chains ingest -> predict -> evaluate
│   └── prediction_pipeline.py  serves single-image predictions
├── src/prayaas/            core package
│   ├── db.py               Firestore client
│   ├── auth.py             bcrypt hashing, JWTs, user queries
│   ├── config/             settings from .env
│   └── research/           notebooks: processing, prediction, testing
├── tests/
└── logs/
```

There is no training stage — the model ships pretrained, and the labelled images are used to evaluate it.

## Setup

```bash
cd backend
py -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Use

```bash
python main.py predict --image "artifacts/raw_data/CANCER/001.jpeg"
python main.py evaluate --limit 40     # quick, stratified across both classes
python main.py evaluate                # all 700 images, a few minutes
python main.py serve --reload          # API on :8000, docs at /docs
pytest
```

## API

Every route is served twice: under `/api/...` (what the frontend calls) and at the root (the contract the original Streamlit client and Render's health check use). `/api/health` and `/health` are the same endpoint.

| Method | Path              | Purpose                                          |
| ------ | ----------------- | ------------------------------------------------ |
| POST   | `/api/signup`     | Create an account. 409 if the email is taken      |
| POST   | `/api/login`      | Returns `{token, user}`. 401 on bad credentials   |
| GET    | `/api/me`         | Current user. Requires `Authorization: Bearer`    |
| GET    | `/api/health`     | Liveness, whether weights are present, class list |
| GET    | `/api/metrics`    | Metrics from the last evaluation                  |
| POST   | `/api/predict`    | Score an uploaded image (multipart)               |
| POST   | `/predict/base64` | Legacy contract: `{"file": "<base64>"}`           |
| POST   | `/api/evaluate`   | Score the labelled images in `raw_data`           |

`/evaluate` is unauthenticated and synchronous, and scoring 700 images takes minutes. Put it behind auth or move it to a background worker before deploying. The `/predict` routes are also unauthenticated — only the frontend gates them, so anyone who can reach the API can score images.

## Auth

Accounts and screening history live in **Firestore** (`users/{id}` and `users/{id}/screenings/{id}`). `src/prayaas/db.py` creates the client from a Firebase service-account key; `src/prayaas/auth.py` holds hashing, JWTs, and user queries.

- Passwords are hashed with **bcrypt** and never stored or returned in the clear.
- Login returns a **JWT** (HS256, signed with `SECRET_KEY`, 7-day expiry by default).
- Wrong password and unknown email return the **same** 401 message, so responses cannot be used to enumerate accounts.
- Emails are compared lower-cased, and a user's document ID is a hash of the lower-cased email, so Firestore's atomic `create()` enforces uniqueness — `A@x.com` and `a@x.com` cannot both exist.

`SECRET_KEY` and the Firebase credentials come from `.env`, which is gitignored. Set `FIREBASE_CREDENTIALS_PATH` to your service-account JSON for local dev, or `FIREBASE_CREDENTIALS_JSON` to its text on hosts with no file to point at (Hugging Face Spaces, Render). The key file itself is gitignored too (`*firebase-adminsdk*.json`) — it carries a private key, so if one is ever committed or pasted somewhere shared, delete it in Firebase console → Project settings → Service accounts and generate a new one.

Leave Firestore **and Storage** in locked mode (deny all client access): only this backend reads or writes them, through the Admin SDK, which bypasses security rules. Firebase's default "test mode" rules leave both open to the whole internet.

### Stored images

For signed-in users, `/predict` keeps the upload and its explainability overlays in Firebase Storage (set `FIREBASE_STORAGE_BUCKET`, e.g. `<project-id>.firebasestorage.app`; unset, screenings are still recorded, just without images). Objects mirror the Firestore path of the screening they belong to:

```
users/{uid}/screenings/{id}/original.jpg       the upload, byte for byte (extension follows its real format)
users/{uid}/screenings/{id}/segmentation.png   U-Net lesion overlay   (abnormal results only)
users/{uid}/screenings/{id}/gradcam.png        Grad-CAM overlay       (abnormal results only)
```

The screening document holds the object paths in an `images` map. Uploads run after the response is sent, so a slow or failed upload never delays or costs the user their result. Anonymous predictions store nothing — there is no owner to file a patient's photo under.

Auth needs the database; screening does not. If Firestore is unreachable the app still boots and `/predict` keeps working, while the auth routes fail — the startup log says so.

## Two things worth knowing

**The model needs `tf_keras`, not `keras`.** The architecture was serialized with Keras 2.15. Keras 3 — what current TensorFlow installs — cannot deserialize it and fails with `Could not locate class 'Functional'`. `pipeline/model_loader.py` therefore imports `tf_keras`, the Keras 2 API, which loads it correctly. Keep `tf-keras` in requirements.

**The model has 3 classes; the data has 2 folders.** It outputs *Oral Cancer*, *No Abnormality detected*, and *Oral premalignant lesion* (OPMD), but only CANCER and NON CANCER exist on disk. Scoring therefore has to collapse OPMD into one of the two, and the choice matters:

| `OPMD_IS_CANCER` | Accuracy | CANCER recall |
| ---------------- | -------- | ------------- |
| `true` (default) | 89.0%    | 0.86          |
| `false`          | 78.6%    | 0.71          |

Measured over all 700 images. The default is `true` on evidence: 75 of the 500 CANCER images are predicted OPMD versus 2 of the 200 NON CANCER ones, so OPMD cases were filed under CANCER when this data was labelled. It is also the safer screening default — a premalignant lesion warrants referral, not an all-clear. Flip it in `.env` if your clinical definition differs.

## Preprocessing

Images are resized to 224×224, given per-channel CLAHE, then scaled to `[0, 1]`. The original Streamlit client applied CLAHE before upload and the API did not, so the enhancement now lives server-side where both paths get it. Disable with `APPLY_CLAHE=false`.

## Deploying

`render.yaml` targets this directory (`rootDir: backend`) and starts `uvicorn app:app`. Weights are read from `artifacts/models/`, and pulled from Hugging Face on first boot if absent. Set `CORS_ORIGINS` to the deployed frontend origin.
