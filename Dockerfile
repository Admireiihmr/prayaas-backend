FROM python:3.11-slim

# libgl1/libglib2.0-0: opencv-python-headless still dlopens these at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces containers run as this non-root user; site-packages must live under
# its home directory since /usr is not writable to it.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
