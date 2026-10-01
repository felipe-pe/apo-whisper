# Base CUDA 12 + cuDNN 9 (exigidos pelo ctranslate2 4.5+).
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /usr/local/bin/

WORKDIR /app
# Dependencias primeiro (camada cacheada): so reinstala quando o lock mudar.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# Modelo DENTRO da imagem: o pull do ACR na mesma regiao e muito mais rapido que
# ler 3 GB por SMB a cada cold start. /data (jobs) continua em Azure Files.
ARG WHISPER_MODEL_NAME=large-v3
ENV WHISPER_MODEL_NAME=${WHISPER_MODEL_NAME} \
    MODEL_DIR=/models
RUN /opt/venv/bin/python -c "import os; from faster_whisper import download_model; \
    print('modelo em', download_model(os.environ['WHISPER_MODEL_NAME'], output_dir=os.environ['MODEL_DIR']))" \
    && du -sh /models

COPY app.py ./

ENV PATH=/opt/venv/bin:$PATH \
    DATA_DIR=/data \
    JOBS_DIR=/data/jobs

EXPOSE 8000
# workers=1 e obrigatorio: modelo e fila sao singletons do processo.
CMD ["uvicorn", "app:web", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "75"]
