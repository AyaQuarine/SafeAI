# SafeAI Common V1 — CPU and local-only by default.
# Demo UI is intentionally light: no ffmpeg, no torch/CUDA, no silero.
# Offline/models stage adds FFmpeg + ASR/diarization/VAD when the network allows.
# No model weights or WAV files are baked into the image.

FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
# Minimal OS deps. ffmpeg stays out of the demo image (huge apt graph + fragile mirrors).
RUN set -eux; \
    for i in 1 2 3; do \
      apt-get update && break || sleep 5; \
    done; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
# Install the package without pulling main deps (silero-vad → torch CUDA on Linux).
# Demo needs FastAPI + numpy/scipy/soundfile for contracts and optional file inspect.
RUN python -m pip install --upgrade pip \
 && python -m pip install -e . --no-deps \
 && python -m pip install \
      "numpy>=1.26,<3" \
      "scipy>=1.11" \
      "soundfile>=0.12" \
      "PyYAML>=6.0" \
      "fastapi>=0.110" \
      "uvicorn[standard]>=0.27" \
      "python-multipart>=0.0.9"
ENV SAFEAI_DISABLE_MICROPHONE=1
EXPOSE 8765

FROM base AS ui
CMD ["uvicorn", "safeai.ui.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8765"]

FROM base AS offline
# Heavier deps for Whisper / Community-1 / Silero. Requires a stable network at build time.
RUN set -eux; \
    for i in 1 2 3; do \
      apt-get update && break || sleep 5; \
    done; \
    apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Force CPU torch first so silero-vad does not pull CUDA wheels.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && python -m pip install \
      "sounddevice>=0.4.6" \
      "silero-vad>=5.1" \
      "onnxruntime>=1.17" \
      "faster-whisper>=1.1" \
      "pyannote.audio>=4.0" \
      "huggingface-hub"
ENV HF_HOME=/home/safeai/.cache/huggingface
CMD ["python", "-m", "safeai.pipeline", "--help"]
