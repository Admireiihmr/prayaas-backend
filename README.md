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
│   ├── db.py               Neon PostgreSQL pool + schema
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

Accounts live in **Neon PostgreSQL**. `src/prayaas/db.py` opens a small pooled connection and creates the `users` table on startup; `src/prayaas/auth.py` holds hashing, JWTs, and queries.

- Passwords are hashed with **bcrypt** and never stored or returned in the clear.
- Login returns a **JWT** (HS256, signed with `SECRET_KEY`, 7-day expiry by default).
- Wrong password and unknown email return the **same** 401 message, so responses cannot be used to enumerate accounts.
- Emails are compared lower-cased, and a `LOWER(email)` unique index enforces that in the database too — `A@x.com` and `a@x.com` cannot both exist.

`DATABASE_URL` and `SECRET_KEY` come from `.env`, which is gitignored. Never hardcode them. If a connection string is ever pasted somewhere shared, rotate it in the Neon console — the password is right there in the URL.

Auth needs the database; screening does not. If Neon is unreachable the app still boots and `/predict` keeps working, while the auth routes fail — the startup log says so.

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
