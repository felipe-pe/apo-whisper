"""Servico de transcricao (faster-whisper em GPU) para Azure Container Apps.

Processo unico rodando com uvicorn num Container App com GPU serverless
(perfil T4):

- Modelo: singleton de processo, carregado em background no lifespan do FastAPI.
- Jobs: fila em memoria + 1 worker em thread (1 job por vez na GPU).
- Estado: job.json/result.json em disco (Azure Files montado em /data).
- Configuracao e segredos: variaveis de ambiente do Container App.
- Retencao: POST /maintenance/cleanup (protegido por token).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

# =============================================================================
# Build / runtime constants
# =============================================================================

BUILD_VERSION = os.getenv("BUILD_VERSION", "").strip() or "azure-aca-whisper-t4-fasterwhisper121-v1"

# Apenas informativo (logs/respostas/custo). Quem escolhe a GPU de verdade e o
# workload profile do Container App (gpu-t4).
GPU_TYPE = os.getenv("GPU_TYPE", "T4").strip() or "T4"

# Modelo + compute configuraveis por env. Na T4 o padrao e float16; se a
# placa/driver recusar, cai para int8_float16 (ver WhisperEngine.load_model).
MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "large-v3").strip() or "large-v3"
COMPUTE_TYPE = (
    os.getenv("WHISPER_COMPUTE_TYPE", "").strip() or os.getenv("COMPUTE_TYPE", "").strip() or "float16"
)
COMPUTE_TYPE_FALLBACK = os.getenv("WHISPER_COMPUTE_TYPE_FALLBACK", "int8_float16").strip() or "int8_float16"

# O modelo vem dentro da imagem, em MODEL_DIR (baixado no build, ver
# Dockerfile). Se MODEL_DIR/model.bin nao existir, o modelo e baixado para
# MODEL_DIR na primeira carga (ver WhisperEngine.load_model).
MODEL_DIR = os.getenv("MODEL_DIR", "/models").strip() or "/models"
# DATA_DIR/JOBS_DIR guardam o estado dos jobs (Azure Files montado em /data).
DATA_DIR = Path(os.getenv("DATA_DIR", "/data").strip() or "/data")
JOBS_DIR = Path(os.getenv("JOBS_DIR", "").strip() or str(DATA_DIR / "jobs"))
IDEMPOTENCY_DIR = Path(os.getenv("IDEMPOTENCY_DIR", "").strip() or str(DATA_DIR / "idempotency"))

MAX_DIRECT_UPLOAD_BYTES = 200 * 1024 * 1024      # 200 MB for multipart uploads
MAX_MEDIA_URL_BYTES = 750 * 1024 * 1024          # 750 MB for URL downloads
CHUNK_SIZE = 1024 * 1024                         # 1 MB
DOWNLOAD_TIMEOUT = (10, 600)                     # (connect, read)
CALLBACK_TIMEOUT = (10, 60)
FFMPEG_TIMEOUT_SEC = 10 * 60
FFPROBE_TIMEOUT_SEC = 60
TRANSCRIBE_TIMEOUT_SEC = 60 * 60

DEFAULT_LANGUAGE = "pt"
DEFAULT_BEAM_SIZE = 5
DEFAULT_PATIENCE = 1.2
DEFAULT_VAD_FILTER = True
DEFAULT_WORD_TIMESTAMPS = False
DEFAULT_SPLIT_TRACKS = True

# Audio pre-processing is now fully server-side. The client sends media as close
# to the original as possible; normalization/loudness is our responsibility.
#   none        -> raw extraction (mono 16k PCM only)
#   light       -> highpass(80Hz) + EBU R128 loudness normalization  [DEFAULT]
#   aggressive  -> light + FFT denoise (afftdn). May remove faint speech.
DEFAULT_AUDIO_PREPROCESS = os.getenv("AUDIO_PREPROCESS_DEFAULT", "light").strip().lower() or "light"
VALID_AUDIO_PREPROCESS = {"none", "light", "aggressive"}

# ---------------------------------------------------------------------------
# Quality-first decoding defaults (anti-hallucination). Velocity is not a
# concern, so we spend compute to maximize transcription quality. All values are
# env-overridable for tuning, but ship with strong defaults.
# ---------------------------------------------------------------------------
DEFAULT_BEST_OF = 5
DEFAULT_COMPRESSION_RATIO_THRESHOLD = 2.4
DEFAULT_LOG_PROB_THRESHOLD = -1.0
DEFAULT_NO_SPEECH_THRESHOLD = 0.6
DEFAULT_HALLUCINATION_SILENCE_THRESHOLD = 2.0
DEFAULT_REPETITION_PENALTY = 1.1
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3
DEFAULT_CONDITION_ON_PREVIOUS_TEXT = False
# Temperature fallback sequence (Whisper default). Enables retry on failed decode.
DEFAULT_TEMPERATURE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# VAD tuned for WhatsApp voice notes (short, quiet, noisy PTT).
DEFAULT_VAD_THRESHOLD = 0.35
DEFAULT_VAD_MIN_SILENCE_MS = 700
DEFAULT_VAD_SPEECH_PAD_MS = 300
DEFAULT_VAD_MIN_SPEECH_MS = 150

# Language detection when `language` is not provided by the client.
DEFAULT_LANGUAGE_DETECTION_SEGMENTS = 3
DEFAULT_LANGUAGE_DETECTION_THRESHOLD = 0.5

# Known PT-BR hallucination boilerplate that Whisper emits over silence/music.
# These segments are removed from the final text and counted for observability.
HALLUCINATION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"legendas?\s+pela\s+comunidade\s+amara",
        r"amara\.org",
        r"legendado\s+por",
        r"legendas?\s+por",
        r"inscreva-?se\s+no\s+canal",
        r"n[aã]o\s+esque[cç]a\s+de\s+se\s+inscrever",
        r"obrigado\s+por\s+assistir",
        r"at[eé]\s+o\s+pr[oó]ximo\s+v[ií]deo",
        r"deixe\s+seu\s+like",
        r"compartilhe\s+esse\s+v[ií]deo",
        r"^\s*legendas?\s*$",
    ]
]
# Segments shorter than this (chars) with high no_speech_prob are suspicious.
HALLUCINATION_SHORT_TEXT_CHARS = 25

# ---------------------------------------------------------------------------
# Media profiles. A profile bundles ffmpeg preprocessing + VAD presets tuned for
# a kind of audio. "auto" picks whatsapp/meeting from the probed duration.
# Anything the client sends explicitly still wins over the preset.
# ---------------------------------------------------------------------------
DEFAULT_MEDIA_PROFILE = os.getenv("DEFAULT_MEDIA_PROFILE", "auto").strip().lower() or "auto"
# Media longer than this is treated as a meeting, shorter as a voice note.
AUTO_PROFILE_DURATION_SPLIT_SEC = float(os.getenv("AUTO_PROFILE_DURATION_SPLIT_SEC", "180"))
LOUDNORM_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"
MEDIA_PROFILES: dict[str, dict[str, Any]] = {
    # No preprocessing at all: raw mono 16k extraction.
    "raw": {
        "audio_enhance": False,
        "denoise": False,
        "highpass_hz": 0,
        "loudness_filter": "",
        "vad_threshold": DEFAULT_VAD_THRESHOLD,
        "vad_min_silence_ms": DEFAULT_VAD_MIN_SILENCE_MS,
        "vad_speech_pad_ms": DEFAULT_VAD_SPEECH_PAD_MS,
        "condition_on_previous_text": DEFAULT_CONDITION_ON_PREVIOUS_TEXT,
    },
    # Short, quiet, noisy PTT. Loudness is the #1 fix; VAD stays sensitive.
    "whatsapp": {
        "audio_enhance": True,
        "denoise": False,
        "highpass_hz": 80,
        "loudness_filter": LOUDNORM_FILTER,
        "vad_threshold": 0.35,
        "vad_min_silence_ms": 700,
        "vad_speech_pad_ms": 300,
        "condition_on_previous_text": False,
    },
    # Long-form, multi-speaker, room noise. Longer silences are natural pauses,
    # so VAD is less trigger-happy and the highpass cuts more rumble.
    "meeting": {
        "audio_enhance": True,
        "denoise": False,
        "highpass_hz": 100,
        "loudness_filter": LOUDNORM_FILTER,
        "vad_threshold": 0.45,
        "vad_min_silence_ms": 1200,
        "vad_speech_pad_ms": 200,
        "condition_on_previous_text": False,
    },
}

# Auth is disabled while TRANSCRIBE_API_TOKEN is empty. If exposed, set it via
# environment (Container App secret) and send Authorization: Bearer <token>.
API_TOKEN_ENV = "TRANSCRIBE_API_TOKEN"
DEFAULT_CALLBACK_URL_ENV = "DEFAULT_CALLBACK_URL"
# Valores padrao de initial_prompt/hotwords quando o request nao envia.
DEFAULT_INITIAL_PROMPT_ENV = "WHISPER_INITIAL_PROMPT"
DEFAULT_HOTWORDS_ENV = "WHISPER_HOTWORDS"

# Statuses that mean "the job is finished and a usable result exists".
COMPLETED_STATUSES = {"completed", "done", "success", "succeeded", "finished"}

# =============================================================================
# App FastAPI
# =============================================================================
# - Imagem: ver Dockerfile (nvidia/cuda + ffmpeg + dependencias do uv.lock).
# - Volume: JOBS_DIR/IDEMPOTENCY_DIR sao pastas comuns em /data (Azure Files).
# - Secrets: tudo vem de variaveis de ambiente do Container App.


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup: sobe o carregamento do modelo, o worker da GPU e a recuperacao
    # de jobs pendentes. Ver secao "Fila em memoria".
    start_background_services()
    try:
        yield
    finally:
        stop_background_services()


web = FastAPI(title="APO Whisper", version=BUILD_VERSION, lifespan=lifespan)

# =============================================================================
# Generic helpers
# =============================================================================


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def new_request_id() -> str:
    return f"req_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:10]}"


def new_job_id() -> str:
    return f"tr_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:10]}"


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "sim", "s", "on"}


def parse_int(value: Any, default: int) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(str(value).strip())
    except Exception:
        return default


def parse_float(value: Any, default: float) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(str(value).strip())
    except Exception:
        return default


# =============================================================================
# Cost / webhook / retention / observability dials (env-driven)
# =============================================================================
# All operational cost knobs live here so behavior can be tuned by changing the
# Container App env vars + new revision, WITHOUT touching code.

# --- GPU cost dials ----------------------------------------------------------
# No Azure quem escala e o Container App (min/max replicas, cooldownPeriod no
# deploy.sh). Aqui estes valores sao so informativos para os eventos de billing;
# mantenha-os iguais ao que foi configurado no deploy.
GPU_SCALEDOWN_WINDOW_SEC = parse_int(os.getenv("GPU_SCALEDOWN_WINDOW_SEC"), 300)
GPU_MIN_CONTAINERS = parse_int(os.getenv("GPU_MIN_CONTAINERS"), 0)
GPU_MAX_CONTAINERS = parse_int(os.getenv("GPU_MAX_CONTAINERS"), 1)

# Estimated USD/hour for the active GPU_TYPE, used only to estimate
# billable_gpu_seconds -> cost in logs/metrics. Override per GPU as needed.
GPU_HOURLY_USD = parse_float(os.getenv("GPU_HOURLY_USD"), 1.10)

# --- Webhook dials -----------------------------------------------------------
ENABLE_WEBHOOK_INGEST = parse_bool(os.getenv("ENABLE_WEBHOOK_INGEST"), False)
CALLBACK_MAX_RETRIES = parse_int(os.getenv("CALLBACK_MAX_RETRIES"), 3)
CALLBACK_BACKOFF_BASE_SEC = parse_float(os.getenv("CALLBACK_BACKOFF_BASE_SEC"), 2.0)
CALLBACK_SIGNING_SECRET = os.getenv("CALLBACK_SIGNING_SECRET", "").strip()
CALLBACK_SCHEMA_VERSION = "1.0"

# --- Retention / cleanup dials -----------------------------------------------
JOB_RETENTION_HOURS = parse_int(os.getenv("JOB_RETENTION_HOURS"), 168)
DELETE_INPUT_AFTER_SUCCESS = parse_bool(os.getenv("DELETE_INPUT_AFTER_SUCCESS"), True)
# A limpeza roda via POST /maintenance/cleanup (chame de um agendador externo,
# ex.: Logic App, GitHub Actions ou cron do Atlas).

# --- Recuperacao de jobs ------------------------------------------------------
# A fila e em memoria; se a replica reiniciar (deploy, scale-to-zero no meio de
# um job), os jobs que ficaram "queued"/"processing" no disco sao reenfileirados
# no startup. So considera jobs mais novos que RECOVER_MAX_AGE_HOURS.
RECOVER_PENDING_JOBS = parse_bool(os.getenv("RECOVER_PENDING_JOBS"), True)
RECOVER_MAX_AGE_HOURS = parse_int(os.getenv("RECOVER_MAX_AGE_HOURS"), 24)

# --- Observability dials -----------------------------------------------------
EMIT_BILLING_EVENTS = parse_bool(os.getenv("EMIT_BILLING_EVENTS"), True)
ESTIMATE_GPU_COST = parse_bool(os.getenv("ESTIMATE_GPU_COST"), True)


def coerce_trace_context(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): str(v)[:500] for k, v in value.items()}
    raw = str(value).strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): str(v)[:500] for k, v in parsed.items()}
    except Exception:
        pass
    return {"_raw": raw[:1000]}


def sanitize_filename(name: str | None, fallback: str = "input_media") -> str:
    if not name:
        return fallback
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    return name[:160] or fallback


def safe_url_info(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "host": parsed.netloc,
        "path_tail": parsed.path[-120:],
        "has_query": bool(parsed.query),
        "sha256_12": hashlib.sha256(url.encode("utf-8")).hexdigest()[:12],
    }


def sha256_12_for_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def log_event(event: str, *, started_at_ms: int | None = None, **fields: Any) -> None:
    payload = {
        "event": event,
        "build_version": BUILD_VERSION,
        "ts": utc_now_iso(),
        **fields,
    }
    if started_at_ms is not None:
        payload["elapsed_sec"] = round((monotonic_ms() - started_at_ms) / 1000, 3)
    print(json.dumps(payload, default=str, ensure_ascii=False), flush=True)


def require_auth(request: Request) -> None:
    expected = os.getenv(API_TOKEN_ENV, "").strip()
    if not expected:
        return
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="unauthorized")


def job_dir(job_id: str) -> Path:
    if not re.match(r"^tr_[A-Za-z0-9T_]+$", job_id):
        raise HTTPException(status_code=400, detail="invalid job_id")
    return JOBS_DIR / job_id


def job_json_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


def result_json_path(job_id: str) -> Path:
    return job_dir(job_id) / "result.json"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def commit_volume_safely() -> None:
    # No-op: /data e um Azure Files montado (SMB) e a escrita ja fica visivel
    # ao fechar o arquivo. Marca os pontos onde o estado do job precisa estar
    # persistido, caso o armazenamento passe a exigir commit explicito.
    return None


def reload_volume_safely() -> None:
    # No-op pelo mesmo motivo do commit_volume_safely().
    return None


def update_job(job_id: str, **updates: Any) -> dict[str, Any]:
    reload_volume_safely()
    path = job_json_path(job_id)
    current = read_json(path) if path.exists() else {"job_id": job_id}
    current.update(updates)
    current["updated_at"] = utc_now_iso()
    atomic_write_json(path, current)
    commit_volume_safely()
    return current


def idempotency_path(request_id: str) -> Path:
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return IDEMPOTENCY_DIR / f"{digest}.json"


def lookup_idempotent_job(request_id: str | None) -> str | None:
    if not request_id:
        return None
    reload_volume_safely()
    path = idempotency_path(request_id)
    if not path.exists():
        return None
    try:
        existing_job_id = read_json(path).get("job_id")
    except Exception:
        return None
    if not existing_job_id:
        return None
    # Only honor the dedup if the referenced job still exists.
    if job_json_path(existing_job_id).exists():
        return existing_job_id
    return None


def store_idempotent_job(request_id: str | None, job_id: str) -> None:
    if not request_id:
        return
    try:
        atomic_write_json(
            idempotency_path(request_id),
            {"request_id": request_id, "job_id": job_id, "created_at": utc_now_iso()},
        )
        commit_volume_safely()
    except Exception as exc:
        log_event("idempotency_store_failed", request_id=request_id, job_id=job_id, error=str(exc)[:500])


def build_options(raw: dict[str, Any]) -> dict[str, Any]:
    audio_preprocess = str(raw.get("audio_preprocess") or DEFAULT_AUDIO_PREPROCESS).strip().lower()
    if audio_preprocess not in VALID_AUDIO_PREPROCESS:
        audio_preprocess = DEFAULT_AUDIO_PREPROCESS

    initial_prompt = (
        str(raw.get("initial_prompt") or os.getenv(DEFAULT_INITIAL_PROMPT_ENV, "")).strip() or None
    )
    hotwords = str(raw.get("hotwords") or os.getenv(DEFAULT_HOTWORDS_ENV, "")).strip() or None

    media_profile = str(raw.get("media_profile") or DEFAULT_MEDIA_PROFILE).strip().lower()
    if media_profile not in set(MEDIA_PROFILES) | {"auto"}:
        media_profile = DEFAULT_MEDIA_PROFILE

    def explicit(key: str) -> bool:
        """True only if the client actually sent this key with a value.

        Keys governed by the media profile MUST stay None when absent, otherwise
        resolve_media_profile() sees a value and the preset never applies.
        """
        return key in raw and str(raw.get(key)).strip() != ""

    return {
        # Client-facing knobs.
        "language": (str(raw.get("language")).strip() if raw.get("language") else None) or DEFAULT_LANGUAGE,
        "beam_size": parse_int(raw.get("beam_size"), DEFAULT_BEAM_SIZE),
        "patience": parse_float(raw.get("patience"), DEFAULT_PATIENCE),
        "vad_filter": parse_bool(raw.get("vad_filter"), DEFAULT_VAD_FILTER),
        "word_timestamps": parse_bool(raw.get("word_timestamps"), DEFAULT_WORD_TIMESTAMPS),
        "split_tracks": parse_bool(raw.get("split_tracks"), DEFAULT_SPLIT_TRACKS),
        "initial_prompt": initial_prompt,
        "hotwords": hotwords,
        "audio_preprocess": audio_preprocess if explicit("audio_preprocess") else None,
        "media_profile": media_profile,
        "callback_url": (
            str(raw.get("callback_url") or os.getenv(DEFAULT_CALLBACK_URL_ENV, "")).strip()
            or None
        ),
        # Advanced quality knobs (env-defaulted, optionally overridable per request).
        "best_of": parse_int(raw.get("best_of"), parse_int(os.getenv("WHISPER_BEST_OF"), DEFAULT_BEST_OF)),
        "compression_ratio_threshold": parse_float(
            raw.get("compression_ratio_threshold"),
            parse_float(os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD"), DEFAULT_COMPRESSION_RATIO_THRESHOLD),
        ),
        "log_prob_threshold": parse_float(
            raw.get("log_prob_threshold"),
            parse_float(os.getenv("WHISPER_LOG_PROB_THRESHOLD"), DEFAULT_LOG_PROB_THRESHOLD),
        ),
        "no_speech_threshold": parse_float(
            raw.get("no_speech_threshold"),
            parse_float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD"), DEFAULT_NO_SPEECH_THRESHOLD),
        ),
        "hallucination_silence_threshold": parse_float(
            raw.get("hallucination_silence_threshold"),
            parse_float(
                os.getenv("WHISPER_HALLUCINATION_SILENCE_THRESHOLD"),
                DEFAULT_HALLUCINATION_SILENCE_THRESHOLD,
            ),
        ),
        "repetition_penalty": parse_float(
            raw.get("repetition_penalty"),
            parse_float(os.getenv("WHISPER_REPETITION_PENALTY"), DEFAULT_REPETITION_PENALTY),
        ),
        "no_repeat_ngram_size": parse_int(
            raw.get("no_repeat_ngram_size"),
            parse_int(os.getenv("WHISPER_NO_REPEAT_NGRAM_SIZE"), DEFAULT_NO_REPEAT_NGRAM_SIZE),
        ),
        # Governed by the media profile: None here means "let the preset decide".
        "condition_on_previous_text": (
            parse_bool(raw.get("condition_on_previous_text"), DEFAULT_CONDITION_ON_PREVIOUS_TEXT)
            if explicit("condition_on_previous_text")
            else None
        ),
        # VAD tuning (also profile-governed).
        "vad_threshold": (
            parse_float(raw.get("vad_threshold"), DEFAULT_VAD_THRESHOLD)
            if explicit("vad_threshold")
            else None
        ),
        "vad_min_silence_ms": (
            parse_int(raw.get("vad_min_silence_ms"), DEFAULT_VAD_MIN_SILENCE_MS)
            if explicit("vad_min_silence_ms")
            else None
        ),
        "vad_speech_pad_ms": (
            parse_int(raw.get("vad_speech_pad_ms"), DEFAULT_VAD_SPEECH_PAD_MS)
            if explicit("vad_speech_pad_ms")
            else None
        ),
        "vad_min_speech_ms": parse_int(raw.get("vad_min_speech_ms"), DEFAULT_VAD_MIN_SPEECH_MS),
        # Hallucination text filter can be disabled per request if needed.
        "filter_hallucinations": parse_bool(raw.get("filter_hallucinations"), True),
    }


def resolve_media_profile(options: dict[str, Any], media_duration_sec: float | None) -> dict[str, Any]:
    """Resolve the effective profile presets, layering explicit user overrides.

    The "auto" profile is decided from media duration (short => whatsapp,
    long => meeting). Any option the user set explicitly (non-None) wins over
    the profile preset.
    """
    profile_name = options.get("media_profile") or DEFAULT_MEDIA_PROFILE
    if profile_name == "auto":
        if media_duration_sec is not None and media_duration_sec > AUTO_PROFILE_DURATION_SPLIT_SEC:
            resolved_name = "meeting"
        else:
            resolved_name = "whatsapp"
    else:
        resolved_name = profile_name

    preset = dict(MEDIA_PROFILES.get(resolved_name, MEDIA_PROFILES["raw"]))

    def pick(key: str, preset_key: str | None = None) -> Any:
        user_val = options.get(key)
        if user_val is not None:
            return user_val
        return preset.get(preset_key or key)

    # An explicit audio_preprocess from the client overrides the preset's
    # enhancement decision (none / light / aggressive), keeping the documented
    # .env contract working on top of the profile system.
    audio_preprocess = options.get("audio_preprocess")
    audio_enhance = bool(pick("audio_enhance"))
    denoise = bool(pick("denoise"))
    if audio_preprocess == "none":
        audio_enhance, denoise = False, False
    elif audio_preprocess == "light":
        audio_enhance, denoise = True, False
    elif audio_preprocess == "aggressive":
        audio_enhance, denoise = True, True

    return {
        "resolved_profile": resolved_name,
        "requested_profile": profile_name,
        "audio_preprocess": audio_preprocess or ("light" if audio_enhance else "none"),
        "audio_enhance": audio_enhance,
        "denoise": denoise,
        "highpass_hz": int(pick("highpass_hz")),
        "loudness_filter": str(pick("loudness_filter")),
        "vad_min_silence_ms": int(pick("vad_min_silence_ms")),
        "vad_speech_pad_ms": int(pick("vad_speech_pad_ms")),
        "vad_threshold": float(pick("vad_threshold")),
        "condition_on_previous_text": bool(pick("condition_on_previous_text")),
    }


def redacted_callback_info(callback_url: str | None) -> dict[str, Any]:
    if not callback_url:
        return {"enabled": False}
    parsed = urlparse(callback_url)
    return {
        "enabled": True,
        "scheme": parsed.scheme,
        "host": parsed.netloc,
        "path_tail": parsed.path[-80:],
    }

# =============================================================================
# Media ingestion helpers on web/API side
# =============================================================================


async def save_upload_file(upload: UploadFile, target: Path, max_bytes: int) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with target.open("wb") as f:
        while True:
            chunk = await upload.read(CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": "file_too_large",
                        "size_bytes": size,
                        "max_bytes": max_bytes,
                    },
                )
            f.write(chunk)
    if size == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_upload", "message": "uploaded file has 0 bytes"},
        )
    return {
        "filename": sanitize_filename(upload.filename),
        "content_type": upload.content_type or "application/octet-stream",
        "size_bytes": size,
        "sha256_12": sha256_12_for_file(target),
    }


def download_media_url(media_url: str, target: Path, max_bytes: int, request_id: str, started_at_ms: int) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    url_info = safe_url_info(media_url)
    log_event("media_url_download_start", started_at_ms=started_at_ms, request_id=request_id, url_info=url_info)

    size = 0
    headers = {"User-Agent": "apo-whisper/1.0", "Accept": "*/*"}
    try:
        with requests.get(media_url, stream=True, timeout=DOWNLOAD_TIMEOUT, headers=headers) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "application/octet-stream")
            declared_length = response.headers.get("Content-Length")
            if declared_length and int(declared_length) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": "file_too_large_declared",
                        "content_length": int(declared_length),
                        "max_bytes": max_bytes,
                    },
                )
            with target.open("wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail={
                                "error": "file_too_large",
                                "downloaded_bytes": size,
                                "max_bytes": max_bytes,
                            },
                        )
                    f.write(chunk)
    except HTTPException:
        raise
    except requests.Timeout as exc:
        raise HTTPException(status_code=504, detail={"error": "download_timeout", "details": str(exc)})
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail={"error": "download_failed", "details": str(exc)[:1000]})

    if size == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_download", "message": "downloaded media has 0 bytes"},
        )

    detected_ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ""
    filename = sanitize_filename(Path(urlparse(media_url).path).name or f"media{detected_ext}")
    info = {
        "filename": filename,
        "content_type": content_type,
        "size_bytes": size,
        "sha256_12": sha256_12_for_file(target),
        "url_info": url_info,
    }
    log_event("media_url_download_done", started_at_ms=started_at_ms, request_id=request_id, input=info)
    return info


async def ingest_request_media(
    request: Request, job_id: str, request_id: str, started_at_ms: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return (input_info, options, extra).

    Saves input media to /data/jobs/<job_id>/input_media. `extra` carries the
    request_id read from the form/body (used for idempotency resolution).
    """
    content_type = request.headers.get("content-type", "")
    target = job_dir(job_id) / "input_media"

    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        # NOTE: request.form() (Starlette) yields starlette.datastructures.UploadFile,
        # which is the *base* class of fastapi.UploadFile. An isinstance() check
        # against fastapi.UploadFile therefore fails for a genuine upload. Use
        # duck-typing (must expose filename + async read) and reject plain strings.
        if (
            upload is None
            or isinstance(upload, str)
            or not hasattr(upload, "read")
            or not hasattr(upload, "filename")
        ):
            raise HTTPException(status_code=400, detail="missing multipart field: file")

        raw_options = {k: v for k, v in form.items() if k != "file"}
        options = build_options(raw_options)
        input_info = await save_upload_file(upload, target, MAX_DIRECT_UPLOAD_BYTES)
        input_info["source"] = "multipart"
        extra = {"form_request_id": str(form.get("request_id") or "").strip() or None}
        return input_info, options, extra

    if "application/json" in content_type:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="json body must be an object")
        media_url = body.get("mediaUrl") or body.get("media_url")
        if not media_url:
            raise HTTPException(status_code=400, detail="missing mediaUrl/media_url")
        options = build_options(body)
        input_info = download_media_url(str(media_url), target, MAX_MEDIA_URL_BYTES, request_id, started_at_ms)
        input_info["source"] = "mediaUrl"
        extra = {"form_request_id": str(body.get("request_id") or "").strip() or None}
        return input_info, options, extra

    # Raw body debug mode. Useful for quick curl tests with --data-binary.
    size = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        async for chunk in request.stream():
            if not chunk:
                continue
            size += len(chunk)
            if size > MAX_DIRECT_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="raw body too large")
            f.write(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="empty request body")
    options = build_options({})
    input_info = {
        "source": "raw_body",
        "filename": "raw_body_media",
        "content_type": content_type or "application/octet-stream",
        "size_bytes": size,
        "sha256_12": sha256_12_for_file(target),
    }
    return input_info, options, {"form_request_id": None}

# =============================================================================
# FFmpeg / faster-whisper helpers on GPU side
# =============================================================================


def run_cmd(cmd: list[str], *, timeout: int, job_id: str, request_id: str, event_prefix: str, started_at_ms: int) -> subprocess.CompletedProcess[bytes]:
    log_event(
        f"{event_prefix}_start",
        started_at_ms=started_at_ms,
        job_id=job_id,
        request_id=request_id,
        cmd=" ".join(cmd[:24]),
        timeout_sec=timeout,
    )
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        log_event(
            f"{event_prefix}_timeout",
            started_at_ms=started_at_ms,
            job_id=job_id,
            request_id=request_id,
            timeout_sec=timeout,
        )
        raise RuntimeError(f"{event_prefix}_timeout: {exc}") from exc

    stderr_tail = result.stderr.decode("utf-8", errors="ignore")[-4000:] if result.stderr else ""
    log_event(
        f"{event_prefix}_done",
        started_at_ms=started_at_ms,
        job_id=job_id,
        request_id=request_id,
        returncode=result.returncode,
        stderr_tail=stderr_tail if result.returncode != 0 else None,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{event_prefix}_failed: {stderr_tail or result.returncode}")
    return result


def build_audio_filters(audio_preprocess: str) -> str | None:
    """Return the ffmpeg -af filter chain for the requested preprocessing level.

    Filters run before the mono/16k/PCM conversion so Whisper receives clean,
    loudness-normalized audio. `loudnorm` (EBU R128) fixes the #1 cause of poor
    WhatsApp PTT transcription: audio that is simply too quiet.
    """
    level = (audio_preprocess or DEFAULT_AUDIO_PREPROCESS).strip().lower()
    if level == "none":
        return None
    loudnorm = "loudnorm=I=-16:TP=-1.5:LRA=11"
    highpass = "highpass=f=80"
    if level == "aggressive":
        # afftdn = FFT-based denoise. Can attenuate faint speech, opt-in only.
        return f"{highpass},afftdn=nf=-25,{loudnorm}"
    # light (default)
    return f"{highpass},{loudnorm}"


def build_audio_filter_chain(profile: dict[str, Any]) -> str | None:
    """Build the ffmpeg -af chain from a resolved media profile.

    Order matters: highpass removes rumble first, denoise works on the cleaned
    signal, and loudness normalization runs last so it measures the final audio.
    """
    if not profile.get("audio_enhance"):
        return None
    parts: list[str] = []
    highpass_hz = int(profile.get("highpass_hz") or 0)
    if highpass_hz > 0:
        parts.append(f"highpass=f={highpass_hz}")
    if profile.get("denoise"):
        # afftdn = FFT-based denoise. Can attenuate faint speech, opt-in only.
        parts.append("afftdn=nf=-25")
    loudness = str(profile.get("loudness_filter") or "").strip()
    if loudness:
        parts.append(loudness)
    return ",".join(parts) or None


def probe_media_duration(probe: dict[str, Any]) -> float | None:
    """Best-effort media duration in seconds from an ffprobe payload.

    Prefers format.duration (whole container) and falls back to the longest
    audio stream. Returns None when nothing usable is present.
    """
    raw = probe.get("raw") or {}
    candidates: list[float] = []
    fmt_duration = (raw.get("format") or {}).get("duration")
    try:
        if fmt_duration is not None:
            candidates.append(float(fmt_duration))
    except (TypeError, ValueError):
        pass
    for stream in probe.get("audio_streams") or []:
        try:
            if stream.get("duration") is not None:
                candidates.append(float(stream["duration"]))
        except (TypeError, ValueError):
            continue
    positive = [c for c in candidates if c > 0]
    return max(positive) if positive else None


def _wav_tail_args(output_path: Path) -> list[str]:
    """Common trailing ffmpeg args that force mono 16 kHz PCM s16le WAV."""
    return [
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]


def ffprobe_audio_streams(input_path: Path, *, job_id: str, request_id: str, started_at_ms: int) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(input_path),
    ]
    result = run_cmd(cmd, timeout=FFPROBE_TIMEOUT_SEC, job_id=job_id, request_id=request_id, event_prefix="ffprobe", started_at_ms=started_at_ms)
    payload = json.loads(result.stdout.decode("utf-8", errors="replace") or "{}")
    audio_streams = [
        s
        for s in payload.get("streams", [])
        if s.get("codec_type") == "audio"
        # Ignore attached_pic (video thumbnail) which is never audio anyway, but
        # guard defensively against odd containers.
        and (s.get("disposition", {}) or {}).get("attached_pic", 0) != 1
    ]
    log_event(
        "ffprobe_audio_streams_detected",
        started_at_ms=started_at_ms,
        job_id=job_id,
        request_id=request_id,
        streams_count=len(audio_streams),
        streams=[
            {
                "index": s.get("index"),
                "codec_name": s.get("codec_name"),
                "channels": s.get("channels"),
                "duration": s.get("duration"),
                "tags": s.get("tags", {}),
            }
            for s in audio_streams
        ],
    )
    return {"raw": payload, "audio_streams": audio_streams}


def extract_stream_to_wav(
    input_path: Path,
    output_path: Path,
    stream_index: int,
    audio_filters: str | None,
    *,
    job_id: str,
    request_id: str,
    started_at_ms: int,
) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        f"0:{stream_index}",
    ]
    if audio_filters:
        cmd += ["-af", audio_filters]
    cmd += _wav_tail_args(output_path)
    run_cmd(
        cmd,
        timeout=FFMPEG_TIMEOUT_SEC,
        job_id=job_id,
        request_id=request_id,
        event_prefix=f"ffmpeg_extract_stream_{stream_index}",
        started_at_ms=started_at_ms,
    )


def normalize_single_audio(
    input_path: Path,
    output_path: Path,
    audio_filters: str | None,
    *,
    job_id: str,
    request_id: str,
    started_at_ms: int,
) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
    ]
    if audio_filters:
        cmd += ["-af", audio_filters]
    cmd += _wav_tail_args(output_path)
    run_cmd(
        cmd,
        timeout=FFMPEG_TIMEOUT_SEC,
        job_id=job_id,
        request_id=request_id,
        event_prefix="ffmpeg_normalize",
        started_at_ms=started_at_ms,
    )


def safe_join_segments_text(parts: list[str]) -> str:
    cleaned = [p.strip() for p in parts if p and p.strip()]
    text = " ".join(cleaned)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def looks_like_hallucination(text: str, no_speech_prob: float) -> bool:
    """Heuristic: known PT-BR boilerplate, or very short text over likely silence."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    for pattern in HALLUCINATION_PATTERNS:
        if pattern.search(stripped):
            return True
    if len(stripped) <= HALLUCINATION_SHORT_TEXT_CHARS and no_speech_prob >= 0.6:
        return True
    return False


def build_vad_parameters(options: dict[str, Any], profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """VAD params: explicit client option > resolved profile preset > default."""
    profile = profile or {}

    def pick(key: str, default: Any) -> Any:
        value = options.get(key)
        if value is None:
            value = profile.get(key)
        return default if value is None else value

    return {
        "threshold": float(pick("vad_threshold", DEFAULT_VAD_THRESHOLD)),
        "min_silence_duration_ms": int(pick("vad_min_silence_ms", DEFAULT_VAD_MIN_SILENCE_MS)),
        "speech_pad_ms": int(pick("vad_speech_pad_ms", DEFAULT_VAD_SPEECH_PAD_MS)),
        "min_speech_duration_ms": int(pick("vad_min_speech_ms", DEFAULT_VAD_MIN_SPEECH_MS)),
    }


def build_transcribe_kwargs(
    options: dict[str, Any], profile: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Quality-first transcribe kwargs. Velocity is not a concern here."""
    profile = profile or {}
    condition_on_previous_text = options.get("condition_on_previous_text")
    if condition_on_previous_text is None:
        condition_on_previous_text = profile.get(
            "condition_on_previous_text", DEFAULT_CONDITION_ON_PREVIOUS_TEXT
        )

    kwargs: dict[str, Any] = {
        "beam_size": int(options.get("beam_size", DEFAULT_BEAM_SIZE)),
        "best_of": int(options.get("best_of", DEFAULT_BEST_OF)),
        "patience": float(options.get("patience", DEFAULT_PATIENCE)),
        "temperature": DEFAULT_TEMPERATURE,
        "compression_ratio_threshold": float(
            options.get("compression_ratio_threshold", DEFAULT_COMPRESSION_RATIO_THRESHOLD)
        ),
        "log_prob_threshold": float(options.get("log_prob_threshold", DEFAULT_LOG_PROB_THRESHOLD)),
        "no_speech_threshold": float(options.get("no_speech_threshold", DEFAULT_NO_SPEECH_THRESHOLD)),
        "repetition_penalty": float(options.get("repetition_penalty", DEFAULT_REPETITION_PENALTY)),
        "no_repeat_ngram_size": int(options.get("no_repeat_ngram_size", DEFAULT_NO_REPEAT_NGRAM_SIZE)),
        "condition_on_previous_text": bool(condition_on_previous_text),
        "word_timestamps": bool(options.get("word_timestamps", DEFAULT_WORD_TIMESTAMPS)),
        "vad_filter": bool(options.get("vad_filter", DEFAULT_VAD_FILTER)),
    }

    # hallucination_silence_threshold requires word_timestamps in faster-whisper.
    hst = options.get("hallucination_silence_threshold")
    if hst and float(hst) > 0 and kwargs["word_timestamps"]:
        kwargs["hallucination_silence_threshold"] = float(hst)

    if kwargs["vad_filter"]:
        kwargs["vad_parameters"] = build_vad_parameters(options, profile)

    language = options.get("language")
    if language:
        kwargs["language"] = language
    else:
        # Auto-detect: gather more evidence before committing to a language.
        kwargs["language_detection_segments"] = DEFAULT_LANGUAGE_DETECTION_SEGMENTS
        kwargs["language_detection_threshold"] = DEFAULT_LANGUAGE_DETECTION_THRESHOLD

    if options.get("initial_prompt"):
        kwargs["initial_prompt"] = options["initial_prompt"]
    if options.get("hotwords"):
        kwargs["hotwords"] = options["hotwords"]
    return kwargs

# =============================================================================
# GPU engine
# =============================================================================


class WhisperEngine:
    """Singleton de processo com o modelo na GPU.

    Uma unica instancia (ENGINE) e criada no import; o modelo e carregado em
    background no startup (lifespan) e todo uso da GPU passa pela fila em
    memoria, entao so um job/warmup roda por vez.
    """

    def __init__(self) -> None:
        self.model = None
        self.compute_type: str | None = None
        self.load_error: str | None = None
        self._ready = threading.Event()
        self._load_lock = threading.Lock()
        self._container_start_ms: int | None = None
        self._cold_start_sec: float | None = None

    def is_ready(self) -> bool:
        return self._ready.is_set() and self.model is not None

    def ensure_loaded(self) -> None:
        """Carrega o modelo se ainda nao estiver carregado (idempotente).

        Usado pela thread de startup e, se ela falhou, de novo pelo worker antes
        de cada job (nova tentativa em vez de travar a fila).
        """
        if self.is_ready():
            return
        with self._load_lock:
            if self.is_ready():
                return
            self.load_model()

    def load_model(self):
        from faster_whisper import WhisperModel

        # Billing observability: mark the wall-clock start of this GPU replica
        # so on_shutdown can report the total billable window. Also measure the
        # cold-start cost (model load) separately from transcription time.
        if self._container_start_ms is None:
            self._container_start_ms = monotonic_ms()
            if EMIT_BILLING_EVENTS:
                log_event(
                    "gpu_container_start",
                    gpu=GPU_TYPE,
                    model=MODEL_NAME,
                    scaledown_window_sec=GPU_SCALEDOWN_WINDOW_SEC,
                    min_containers=GPU_MIN_CONTAINERS,
                    max_containers=GPU_MAX_CONTAINERS,
                )

        started_at_ms = monotonic_ms()
        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        # O modelo vem dentro da imagem: o build (Dockerfile) grava os arquivos
        # direto em MODEL_DIR. Se MODEL_DIR/model.bin existir, carrega dali sem
        # baixar nada. Sem ele (ex.: execucao fora da imagem), o faster-whisper
        # baixa o modelo para MODEL_DIR agora e reaproveita nas proximas cargas.
        model_in_image = (Path(MODEL_DIR) / "model.bin").is_file()
        model_cached = model_in_image or any(
            Path(MODEL_DIR).glob(f"models--*faster-whisper-{MODEL_NAME}*")
        )
        log_event(
            "model_load_start",
            started_at_ms=started_at_ms,
            model=MODEL_NAME,
            gpu=GPU_TYPE,
            compute_type=COMPUTE_TYPE,
            model_dir=MODEL_DIR,
            model_in_image=model_in_image,
            model_cached=model_cached,
        )

        # float16 primeiro (padrao na T4); se falhar, tenta o fallback.
        compute_types = [COMPUTE_TYPE]
        if COMPUTE_TYPE_FALLBACK and COMPUTE_TYPE_FALLBACK != COMPUTE_TYPE:
            compute_types.append(COMPUTE_TYPE_FALLBACK)
        last_exc: Exception | None = None
        for compute_type in compute_types:
            try:
                if model_in_image:
                    self.model = WhisperModel(
                        MODEL_DIR,
                        device="cuda",
                        compute_type=compute_type,
                    )
                else:
                    self.model = WhisperModel(
                        MODEL_NAME,
                        device="cuda",
                        compute_type=compute_type,
                        download_root=MODEL_DIR,
                    )
                self.compute_type = compute_type
                break
            except Exception as exc:
                last_exc = exc
                log_event(
                    "model_load_compute_type_failed",
                    started_at_ms=started_at_ms,
                    model=MODEL_NAME,
                    compute_type=compute_type,
                    error=str(exc)[:1000],
                )
        if self.model is None:
            self.load_error = str(last_exc)[:2000] if last_exc else "unknown_error"
            log_event("model_load_failed", started_at_ms=started_at_ms, model=MODEL_NAME, error=self.load_error)
            raise RuntimeError(f"model_load_failed: {self.load_error}")

        self.load_error = None
        self._cold_start_sec = round((monotonic_ms() - started_at_ms) / 1000, 3)
        self._ready.set()
        log_event(
            "model_load_done",
            started_at_ms=started_at_ms,
            model=MODEL_NAME,
            gpu=GPU_TYPE,
            compute_type=self.compute_type,
            cold_start_sec=self._cold_start_sec,
        )

    def on_shutdown(self):
        # Chamado no shutdown do uvicorn (o Container App derrubando a replica
        # apos o cooldown). Emits the total billable GPU window for cost tracking.
        if not EMIT_BILLING_EVENTS:
            return
        start = getattr(self, "_container_start_ms", None)
        if start is None:
            return
        alive_sec = round((monotonic_ms() - start) / 1000, 3)
        fields: dict[str, Any] = {
            "gpu": GPU_TYPE,
            "container_alive_sec": alive_sec,
            "scaledown_window_sec": GPU_SCALEDOWN_WINDOW_SEC,
        }
        if ESTIMATE_GPU_COST:
            fields["estimated_cost_usd"] = round(alive_sec / 3600.0 * GPU_HOURLY_USD, 4)
        log_event("gpu_container_idle_shutdown", **fields)

    def warmup(self) -> dict[str, Any]:
        started_at_ms = monotonic_ms()
        log_event("warmup_gpu_start", started_at_ms=started_at_ms, model=MODEL_NAME, gpu=GPU_TYPE)
        # Run a tiny synthetic transcription to force CUDA/cuDNN kernel
        # compilation now, so the first real job runs on the hot path.
        synthetic_ok = False
        synthetic_error: str | None = None
        try:
            import numpy as np

            silence = np.zeros(16000, dtype=np.float32)  # 1s of silence @ 16kHz
            segments, _info = self.model.transcribe(
                silence,
                language="pt",
                beam_size=1,
                vad_filter=False,
            )
            for _ in segments:  # consume generator to actually run inference
                pass
            synthetic_ok = True
        except Exception as exc:  # pragma: no cover - defensive
            synthetic_error = str(exc)[:1000]
            log_event("warmup_synthetic_failed", started_at_ms=started_at_ms, error=synthetic_error)

        ready = self.model is not None
        result = {
            "ok": True,
            "status": "ready" if ready else "not_ready",
            "build_version": BUILD_VERSION,
            "model": MODEL_NAME,
            "gpu": GPU_TYPE,
            "device": "cuda",
            "compute_type": self.compute_type or COMPUTE_TYPE,
            "synthetic_inference_ok": synthetic_ok,
            "synthetic_error": synthetic_error,
            "elapsed_sec": round((monotonic_ms() - started_at_ms) / 1000, 3),
        }
        log_event("warmup_gpu_done", started_at_ms=started_at_ms, **result)
        return result

    def process_job(self, job_id: str) -> dict[str, Any]:
        started_at_ms = monotonic_ms()
        reload_volume_safely()

        job_path = job_json_path(job_id)
        if not job_path.exists():
            raise RuntimeError(f"job_not_found: {job_id}")

        job = read_json(job_path)
        request_id = job.get("request_id") or new_request_id()
        options = job.get("options") or {}
        input_info = job.get("input") or {}
        input_path = job_dir(job_id) / "input_media"

        # Time spent queued before the GPU picked up the job.
        created_at = job.get("created_at")
        queue_wait_sec = None
        try:
            if created_at:
                created_dt = datetime.fromisoformat(created_at)
                queue_wait_sec = round((datetime.now(timezone.utc) - created_dt).total_seconds(), 3)
        except Exception:
            queue_wait_sec = None

        log_event(
            "job_processing_start",
            started_at_ms=started_at_ms,
            job_id=job_id,
            request_id=request_id,
            queue_wait_sec=queue_wait_sec,
            input=input_info,
            options={k: v for k, v in options.items() if k not in {"initial_prompt"}},
            initial_prompt_chars=len(options.get("initial_prompt") or ""),
            callback=redacted_callback_info(options.get("callback_url")),
        )
        update_job(
            job_id,
            status="processing",
            stage="started",
            processing_started_at=utc_now_iso(),
            queue_wait_sec=queue_wait_sec,
        )

        try:
            if not input_path.exists():
                raise RuntimeError("input_media_not_found")

            with tempfile.TemporaryDirectory() as tmp:
                tmpdir = Path(tmp)
                local_input = tmpdir / sanitize_filename(input_info.get("filename"), "input_media")
                shutil.copyfile(input_path, local_input)

                update_job(job_id, stage="ffprobe")
                probe = ffprobe_audio_streams(local_input, job_id=job_id, request_id=request_id, started_at_ms=started_at_ms)
                audio_streams = probe["audio_streams"]
                if not audio_streams:
                    raise RuntimeError("no_audio_streams_detected")

                # Resolve the media profile (P1/P2). "auto" is decided from the
                # probed media duration; explicit user options still win.
                media_duration_hint = probe_media_duration(probe)
                profile = resolve_media_profile(options, media_duration_hint)
                audio_filters = build_audio_filter_chain(profile)
                log_event(
                    "media_profile_resolved",
                    started_at_ms=started_at_ms,
                    job_id=job_id,
                    request_id=request_id,
                    media_duration_hint=media_duration_hint,
                    profile=profile,
                    audio_filters=audio_filters or "(none)",
                )
                update_job(job_id, stage="profile_resolved", profile=profile, audio_filters=audio_filters or None)

                split_tracks = bool(options.get("split_tracks", DEFAULT_SPLIT_TRACKS))
                audio_preprocess = str(profile.get("audio_preprocess") or DEFAULT_AUDIO_PREPROCESS)
                streams_to_process = audio_streams if split_tracks else [audio_streams[0]]

                tracks: list[dict[str, Any]] = []
                total_segments = 0
                total_text_chars = 0
                total_media_duration = 0.0
                total_hallucinations_removed = 0
                language_probabilities: list[float] = []
                requested_language = options.get("language")
                transcribe_kwargs = build_transcribe_kwargs(options, profile)

                for track_index, stream in enumerate(streams_to_process):
                    stream_index = int(stream["index"])
                    update_job(job_id, stage="extract_track", current_track=track_index, tracks_total=len(streams_to_process))

                    wav_path = tmpdir / f"track_{track_index}_stream_{stream_index}_16khz.wav"
                    log_event(
                        "track_detected",
                        started_at_ms=started_at_ms,
                        job_id=job_id,
                        request_id=request_id,
                        track_index=track_index,
                        stream_index=stream_index,
                        audio_preprocess=audio_preprocess,
                        stream={
                            "codec_name": stream.get("codec_name"),
                            "channels": stream.get("channels"),
                            "duration": stream.get("duration"),
                            "sample_rate": stream.get("sample_rate"),
                            "tags": stream.get("tags", {}),
                        },
                    )

                    if split_tracks:
                        extract_stream_to_wav(
                            local_input,
                            wav_path,
                            stream_index,
                            audio_filters,
                            job_id=job_id,
                            request_id=request_id,
                            started_at_ms=started_at_ms,
                        )
                    else:
                        normalize_single_audio(
                            local_input,
                            wav_path,
                            audio_filters,
                            job_id=job_id,
                            request_id=request_id,
                            started_at_ms=started_at_ms,
                        )

                    update_job(job_id, stage="transcribe_track", current_track=track_index)
                    track_started_ms = monotonic_ms()
                    log_event(
                        "transcribe_track_start",
                        started_at_ms=started_at_ms,
                        job_id=job_id,
                        request_id=request_id,
                        track_index=track_index,
                        stream_index=stream_index,
                        wav_size_bytes=wav_path.stat().st_size,
                        params={k: v for k, v in transcribe_kwargs.items() if k not in {"initial_prompt"}},
                        initial_prompt_chars=len(str(transcribe_kwargs.get("initial_prompt") or "")),
                    )

                    try:
                        segments_iter, info = self.model.transcribe(str(wav_path), **transcribe_kwargs)
                    except TypeError as exc:
                        # Defensive fallback for optional kwargs across faster-whisper versions.
                        fallback_kwargs = dict(transcribe_kwargs)
                        fallback_kwargs.pop("hotwords", None)
                        fallback_kwargs.pop("patience", None)
                        log_event(
                            "transcribe_kwargs_fallback",
                            started_at_ms=started_at_ms,
                            job_id=job_id,
                            request_id=request_id,
                            track_index=track_index,
                            error=str(exc)[:1000],
                            fallback_keys=sorted(fallback_kwargs.keys()),
                        )
                        segments_iter, info = self.model.transcribe(str(wav_path), **fallback_kwargs)

                    text_parts: list[str] = []
                    segment_list: list[dict[str, Any]] = []
                    logprob_sum = 0.0
                    logprob_weight = 0.0
                    no_speech_probs: list[float] = []
                    compression_ratios: list[float] = []
                    hallucinations_removed = 0

                    for seg in segments_iter:
                        seg_text = (seg.text or "").strip()
                        if not seg_text:
                            continue
                        seg_no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
                        seg_avg_logprob = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
                        seg_compression = float(getattr(seg, "compression_ratio", 0.0) or 0.0)

                        # Drop known hallucination boilerplate before it reaches the client.
                        if bool(options.get("filter_hallucinations", True)) and looks_like_hallucination(
                            seg_text, seg_no_speech
                        ):
                            hallucinations_removed += 1
                            log_event(
                                "segment_hallucination_filtered",
                                started_at_ms=started_at_ms,
                                job_id=job_id,
                                request_id=request_id,
                                track_index=track_index,
                                text=seg_text[:200],
                                no_speech_prob=round(seg_no_speech, 4),
                            )
                            continue

                        text_parts.append(seg_text)
                        seg_duration = max(float(seg.end) - float(seg.start), 0.0)
                        logprob_sum += seg_avg_logprob * (seg_duration or 1.0)
                        logprob_weight += (seg_duration or 1.0)
                        no_speech_probs.append(seg_no_speech)
                        compression_ratios.append(seg_compression)

                        out: dict[str, Any] = {
                            "start": round(float(seg.start), 2),
                            "end": round(float(seg.end), 2),
                            "text": seg_text,
                            "avg_logprob": round(seg_avg_logprob, 4),
                            "no_speech_prob": round(seg_no_speech, 4),
                        }
                        words = getattr(seg, "words", None)
                        if options.get("word_timestamps") and words:
                            out["words"] = [
                                {
                                    "start": round(float(w.start), 2),
                                    "end": round(float(w.end), 2),
                                    "word": w.word,
                                    "probability": round(float(w.probability), 4),
                                }
                                for w in words
                            ]
                        segment_list.append(out)

                    text = safe_join_segments_text(text_parts)
                    duration = float(getattr(info, "duration", 0.0) or 0.0)
                    duration_after_vad = float(getattr(info, "duration_after_vad", 0.0) or 0.0)
                    transcribe_elapsed = round((monotonic_ms() - track_started_ms) / 1000, 3)
                    speed_x = round(duration / transcribe_elapsed, 3) if transcribe_elapsed > 0 else None
                    language = getattr(info, "language", None)
                    language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)

                    # Aggregate quality signals consumed by the client Quality Gate v2.
                    track_avg_logprob = round(logprob_sum / logprob_weight, 4) if logprob_weight > 0 else None
                    track_no_speech_max = round(max(no_speech_probs), 4) if no_speech_probs else None
                    track_no_speech_avg = round(sum(no_speech_probs) / len(no_speech_probs), 4) if no_speech_probs else None
                    track_compression = round(sum(compression_ratios) / len(compression_ratios), 4) if compression_ratios else None
                    language_mismatch = bool(requested_language and language and language != requested_language)

                    quality = {
                        # `avg_logprob` = duration-weighted mean; client takes the worst (min) across tracks.
                        "avg_logprob": track_avg_logprob,
                        # `no_speech_prob` = worst (max) of the track; client penalizes > 0.6 + short text.
                        "no_speech_prob": track_no_speech_max,
                        "avg_no_speech_prob": track_no_speech_avg,
                        "compression_ratio": track_compression,
                        "hallucinations_removed": hallucinations_removed,
                        "language_mismatch": language_mismatch,
                        "suspected_hallucination": bool(
                            len(text) <= HALLUCINATION_SHORT_TEXT_CHARS
                            and (track_no_speech_max or 0.0) >= 0.6
                        ),
                    }

                    track_result = {
                        "track_index": track_index,
                        "stream_index": stream_index,
                        "codec_name": stream.get("codec_name"),
                        "channels": stream.get("channels"),
                        "source_duration": stream.get("duration"),
                        "text": text,
                        "language": language,
                        "language_probability": round(language_probability, 4),
                        "duration_sec": round(duration, 2),
                        "duration_after_vad_sec": round(duration_after_vad, 2),
                        "transcription_elapsed_sec": transcribe_elapsed,
                        "speed_x": speed_x,
                        "segments_count": len(segment_list),
                        "text_chars": len(text),
                        "quality": quality,
                        "segments": segment_list,
                    }
                    tracks.append(track_result)
                    total_segments += len(segment_list)
                    total_text_chars += len(text)
                    total_media_duration += duration
                    total_hallucinations_removed += hallucinations_removed
                    language_probabilities.append(language_probability)

                    log_event(
                        "transcribe_track_done",
                        started_at_ms=started_at_ms,
                        job_id=job_id,
                        request_id=request_id,
                        **{k: v for k, v in track_result.items() if k not in {"segments", "text"}},
                    )

                # Ordered by track_index, only tracks that produced speech, joined by "\n".
                # CRITICAL: no labels/headers inside result.text (goes straight to WhatsApp).
                combined_text = "\n".join(
                    t["text"] for t in sorted(tracks, key=lambda x: x["track_index"]) if t["text"].strip()
                ).strip()

                avg_language_probability = (
                    round(sum(language_probabilities) / len(language_probabilities), 4)
                    if language_probabilities
                    else None
                )

                elapsed_sec = round((monotonic_ms() - started_at_ms) / 1000, 3)
                result = {
                    "ok": True,
                    "event": "transcription.completed",
                    "job_id": job_id,
                    "request_id": request_id,
                    "build_version": BUILD_VERSION,
                    "model": MODEL_NAME,
                    "device": "cuda",
                    "gpu": GPU_TYPE,
                    "status": "completed",
                    "created_at": job.get("created_at"),
                    "completed_at": utc_now_iso(),
                    "elapsed_sec": elapsed_sec,
                    "input": input_info,
                    "options_used": {
                        "language": options.get("language"),
                        "beam_size": options.get("beam_size"),
                        "patience": options.get("patience"),
                        "vad_filter": options.get("vad_filter"),
                        "word_timestamps": options.get("word_timestamps"),
                        "split_tracks": options.get("split_tracks"),
                        "audio_preprocess": audio_preprocess,
                        "best_of": options.get("best_of"),
                        "repetition_penalty": options.get("repetition_penalty"),
                        "no_repeat_ngram_size": options.get("no_repeat_ngram_size"),
                        "has_initial_prompt": bool(options.get("initial_prompt")),
                        "has_hotwords": bool(options.get("hotwords")),
                        # P1/P2 resolved settings
                        "media_profile_requested": options.get("media_profile"),
                        "media_profile_resolved": profile.get("resolved_profile"),
                        "audio_enhance": profile.get("audio_enhance"),
                        "denoise": profile.get("denoise"),
                        "highpass_hz": profile.get("highpass_hz"),
                        "loudness_filter": profile.get("loudness_filter"),
                        "audio_filters": audio_filters or None,
                        "temperature": transcribe_kwargs.get("temperature"),
                        "no_speech_threshold": transcribe_kwargs.get("no_speech_threshold"),
                        "log_prob_threshold": transcribe_kwargs.get("log_prob_threshold"),
                        "compression_ratio_threshold": transcribe_kwargs.get("compression_ratio_threshold"),
                        "hallucination_silence_threshold": transcribe_kwargs.get("hallucination_silence_threshold"),
                        "condition_on_previous_text": transcribe_kwargs.get("condition_on_previous_text"),
                        "vad_parameters": transcribe_kwargs.get("vad_parameters"),
                    },
                    "metrics": {
                        "input_size_bytes": input_info.get("size_bytes"),
                        "media_duration_sec_sum": round(total_media_duration, 2),
                        "elapsed_sec": elapsed_sec,
                        "overall_speed_x": round(total_media_duration / elapsed_sec, 3) if elapsed_sec > 0 else None,
                        "tracks_count": len(tracks),
                        "segments_count": total_segments,
                        "text_chars": total_text_chars,
                        "avg_language_probability": avg_language_probability,
                        "hallucinations_removed": total_hallucinations_removed,
                        # Cost observability: cold start (model load) is separated
                        # from transcription time; billable seconds approximate the
                        # GPU wall-clock consumed by this job.
                        "cold_start_sec": getattr(self, "_cold_start_sec", None),
                        "billable_gpu_seconds": elapsed_sec,
                        "estimated_cost_usd": (
                            round(elapsed_sec / 3600.0 * GPU_HOURLY_USD, 4)
                            if ESTIMATE_GPU_COST
                            else None
                        ),
                    },
                    "result": {
                        "text": combined_text,
                        "tracks": tracks,
                    },
                }

                atomic_write_json(result_json_path(job_id), result)
                update_job(
                    job_id,
                    status="completed",
                    stage="done",
                    result_available=True,
                    completed_at=result["completed_at"],
                    metrics=result["metrics"],
                    result_path=str(result_json_path(job_id)),
                )

                # Retention: drop the heavy input_media right after success to
                # save volume storage. job.json + result.json are kept. Toggle
                # via DELETE_INPUT_AFTER_SUCCESS.
                if DELETE_INPUT_AFTER_SUCCESS:
                    try:
                        if input_path.exists():
                            freed_bytes = input_path.stat().st_size
                            input_path.unlink()
                            log_event(
                                "input_media_deleted",
                                started_at_ms=started_at_ms,
                                job_id=job_id,
                                request_id=request_id,
                                freed_bytes=freed_bytes,
                            )
                    except Exception as cleanup_exc:
                        log_event(
                            "input_media_delete_failed",
                            started_at_ms=started_at_ms,
                            job_id=job_id,
                            request_id=request_id,
                            error=str(cleanup_exc)[:1000],
                        )

                commit_volume_safely()

                log_event(
                    "job_completed",
                    started_at_ms=started_at_ms,
                    job_id=job_id,
                    request_id=request_id,
                    metrics=result["metrics"],
                    language_probabilities=[round(p, 4) for p in language_probabilities],
                )

                callback_url = options.get("callback_url")
                if callback_url:
                    self._send_callback(callback_url, result, job_id, request_id, started_at_ms)

                return result

        except Exception as exc:
            error_payload = {
                "ok": False,
                "event": "transcription.failed",
                "job_id": job_id,
                "request_id": request_id,
                "build_version": BUILD_VERSION,
                "status": "failed",
                "stage": "processing",
                "error": "transcription_failed",
                "details": str(exc)[:4000],
                "failed_at": utc_now_iso(),
                "elapsed_sec": round((monotonic_ms() - started_at_ms) / 1000, 3),
                "input": input_info,
            }
            atomic_write_json(result_json_path(job_id), error_payload)
            update_job(
                job_id,
                status="failed",
                stage="failed",
                result_available=False,
                failed_at=error_payload["failed_at"],
                error=error_payload["details"],
                details=error_payload["details"],
            )
            commit_volume_safely()
            log_event("job_failed", started_at_ms=started_at_ms, **error_payload)
            callback_url = options.get("callback_url")
            if callback_url:
                self._send_callback(callback_url, error_payload, job_id, request_id, started_at_ms)
            return error_payload

    def _send_callback(self, callback_url: str, payload: dict[str, Any], job_id: str, request_id: str, started_at_ms: int) -> None:
        """POST the result to the client webhook with retry/backoff + optional HMAC.

        - Adds `schema_version` to the payload (contract v1) without mutating the
          caller's dict.
        - Retries on network errors and 5xx responses up to CALLBACK_MAX_RETRIES
          with exponential backoff (CALLBACK_BACKOFF_BASE_SEC ** attempt).
        - If CALLBACK_SIGNING_SECRET is set, sends `X-Signature` = HMAC-SHA256 of
          the exact JSON body so the receiver can verify authenticity.
        """
        info = redacted_callback_info(callback_url)
        # Version the payload without mutating the original (aditivo, backward compat).
        signed_payload = {"schema_version": CALLBACK_SCHEMA_VERSION, **payload}
        body = json.dumps(signed_payload, ensure_ascii=False, default=str).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
            "X-Schema-Version": CALLBACK_SCHEMA_VERSION,
        }
        if CALLBACK_SIGNING_SECRET:
            signature = hmac.new(
                CALLBACK_SIGNING_SECRET.encode("utf-8"), body, hashlib.sha256
            ).hexdigest()
            headers["X-Signature"] = f"sha256={signature}"

        log_event(
            "callback_start",
            started_at_ms=started_at_ms,
            job_id=job_id,
            request_id=request_id,
            callback=info,
            max_retries=CALLBACK_MAX_RETRIES,
            signed=bool(CALLBACK_SIGNING_SECRET),
        )

        last_error: str | None = None
        # attempt 0 is the first try; retries go up to CALLBACK_MAX_RETRIES.
        for attempt in range(CALLBACK_MAX_RETRIES + 1):
            try:
                response = requests.post(
                    callback_url,
                    data=body,
                    timeout=CALLBACK_TIMEOUT,
                    headers=headers,
                )
                if response.status_code >= 500:
                    # Server-side error: retryable.
                    last_error = f"http_{response.status_code}"
                    log_event(
                        "callback_retryable_error",
                        started_at_ms=started_at_ms,
                        job_id=job_id,
                        request_id=request_id,
                        attempt=attempt,
                        status_code=response.status_code,
                        response_text_tail=response.text[-500:] if response.text else "",
                    )
                else:
                    log_event(
                        "callback_done",
                        started_at_ms=started_at_ms,
                        job_id=job_id,
                        request_id=request_id,
                        attempt=attempt,
                        status_code=response.status_code,
                        response_text_tail=response.text[-1000:] if response.text else "",
                    )
                    return
            except Exception as exc:
                last_error = str(exc)[:2000]
                log_event(
                    "callback_retryable_error",
                    started_at_ms=started_at_ms,
                    job_id=job_id,
                    request_id=request_id,
                    attempt=attempt,
                    error=last_error,
                )

            # Backoff before the next attempt (skip sleeping after the last one).
            if attempt < CALLBACK_MAX_RETRIES:
                time.sleep(CALLBACK_BACKOFF_BASE_SEC ** attempt)

        log_event(
            "callback_failed",
            started_at_ms=started_at_ms,
            job_id=job_id,
            request_id=request_id,
            callback=info,
            attempts=CALLBACK_MAX_RETRIES + 1,
            error=last_error,
        )

# =============================================================================
# Fila em memoria + worker da GPU
# =============================================================================
# Um unico worker em thread consome a fila, entao a GPU roda um job (ou warmup)
# por vez. O estado do job fica no job.json em disco; a fila so carrega o
# job_id. Cada item leva um Future para os endpoints sincronos (/warmup,
# /transcribe) aguardarem o resultado.

ENGINE = WhisperEngine()
JOB_QUEUE: "queue.Queue[tuple[str, str | None, Future]]" = queue.Queue()
_IN_FLIGHT: set[str] = set()
_IN_FLIGHT_LOCK = threading.Lock()
_STOP = "__stop__"
_WORKER_THREAD: threading.Thread | None = None


def submit_job(job_id: str) -> Future:
    """Enfileira um job para a GPU. Ignora duplicado ja na fila/processando."""
    fut: Future = Future()
    with _IN_FLIGHT_LOCK:
        if job_id in _IN_FLIGHT:
            fut.set_result({"ok": True, "job_id": job_id, "status": "already_queued"})
            return fut
        _IN_FLIGHT.add(job_id)
    JOB_QUEUE.put(("job", job_id, fut))
    return fut


def submit_warmup() -> Future:
    fut: Future = Future()
    JOB_QUEUE.put(("warmup", None, fut))
    return fut


def fail_job_without_engine(job_id: str, details: str) -> dict[str, Any]:
    """Marca o job como failed quando o modelo nao carregou.

    Mesmo formato do error_payload de WhisperEngine.process_job, para o cliente
    (status/result/callback) nao perceber diferenca.
    """
    started_at_ms = monotonic_ms()
    job = read_json(job_json_path(job_id)) if job_json_path(job_id).exists() else {}
    request_id = job.get("request_id") or new_request_id()
    error_payload = {
        "ok": False,
        "event": "transcription.failed",
        "job_id": job_id,
        "request_id": request_id,
        "build_version": BUILD_VERSION,
        "status": "failed",
        "stage": "processing",
        "error": "transcription_failed",
        "details": details[:4000],
        "failed_at": utc_now_iso(),
        "elapsed_sec": 0.0,
        "input": job.get("input") or {},
    }
    atomic_write_json(result_json_path(job_id), error_payload)
    update_job(
        job_id,
        status="failed",
        stage="failed",
        result_available=False,
        failed_at=error_payload["failed_at"],
        error=error_payload["details"],
        details=error_payload["details"],
    )
    log_event("job_failed", started_at_ms=started_at_ms, **error_payload)
    callback_url = (job.get("options") or {}).get("callback_url")
    if callback_url:
        ENGINE._send_callback(callback_url, error_payload, job_id, request_id, started_at_ms)
    return error_payload


def gpu_worker_loop() -> None:
    if RECOVER_PENDING_JOBS:
        try:
            recover_pending_jobs()
        except Exception as exc:
            log_event("recover_pending_jobs_failed", error=str(exc)[:1000])

    while True:
        kind, job_id, fut = JOB_QUEUE.get()
        if kind == _STOP:
            break
        if not fut.set_running_or_notify_cancel():
            continue
        try:
            try:
                ENGINE.ensure_loaded()
            except Exception as exc:
                if kind == "job" and job_id:
                    fut.set_result(fail_job_without_engine(job_id, f"model_not_loaded: {exc}"))
                else:
                    fut.set_exception(exc)
                continue
            if kind == "warmup":
                fut.set_result(ENGINE.warmup())
            else:
                fut.set_result(ENGINE.process_job(str(job_id)))
        except BaseException as exc:  # process_job ja trata seus erros; isto e defensivo
            log_event("gpu_worker_error", kind=kind, job_id=job_id, error=str(exc)[:2000])
            if not fut.done():
                fut.set_exception(exc)
        finally:
            if kind == "job" and job_id:
                with _IN_FLIGHT_LOCK:
                    _IN_FLIGHT.discard(job_id)
            JOB_QUEUE.task_done()


def recover_pending_jobs() -> None:
    """Reenfileira jobs que ficaram queued/processing no disco.

    A fila e em memoria e se perde quando a replica reinicia. Jobs em
    "processing" recomecam do zero (o input_media so e apagado apos sucesso).
    """
    if not JOBS_DIR.exists():
        return
    cutoff = datetime.now(timezone.utc).timestamp() - (RECOVER_MAX_AGE_HOURS * 3600)
    pending: list[tuple[str, str]] = []
    for entry in JOBS_DIR.iterdir():
        jpath = entry / "job.json"
        if not entry.is_dir() or not jpath.exists():
            continue
        try:
            job = read_json(jpath)
            status = str(job.get("status") or "").lower()
            created_at = str(job.get("created_at") or "")
            if status not in {"queued", "processing"}:
                continue
            if not created_at or datetime.fromisoformat(created_at).timestamp() < cutoff:
                continue
            if not (entry / "input_media").exists():
                continue
            pending.append((created_at, entry.name))
        except Exception as exc:
            log_event("recover_job_read_failed", job_id=entry.name, error=str(exc)[:500])
    for _created_at, job_id in sorted(pending):
        submit_job(job_id)
    log_event("recover_pending_jobs_done", recovered_count=len(pending), job_ids=[j for _, j in sorted(pending)][:50])


def _load_model_in_background() -> None:
    try:
        ENGINE.ensure_loaded()
    except Exception as exc:
        # O worker tenta de novo antes do proximo job; aqui so registra.
        log_event("model_background_load_failed", error=str(exc)[:2000])


def start_background_services() -> None:
    """Startup (lifespan): carrega o modelo e sobe o worker da GPU.

    O carregamento roda em thread para a porta 8000 abrir na hora, sem depender
    do tempo de carga do modelo para passar no probe de startup do Container
    App. Requests de warmup/jobs esperam o modelo na fila.
    """
    global _WORKER_THREAD
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    IDEMPOTENCY_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_load_model_in_background, name="model-loader", daemon=True).start()
    _WORKER_THREAD = threading.Thread(target=gpu_worker_loop, name="gpu-worker", daemon=True)
    _WORKER_THREAD.start()
    log_event("background_services_started", model=MODEL_NAME, gpu=GPU_TYPE, jobs_dir=str(JOBS_DIR), model_dir=MODEL_DIR)


def stop_background_services() -> None:
    JOB_QUEUE.put((_STOP, None, Future()))
    ENGINE.on_shutdown()


# =============================================================================
# FastAPI endpoints
# =============================================================================


@web.get("/health")
async def health(request: Request, deep: bool = False):
    require_auth(request)
    started_at_ms = monotonic_ms()
    request_id = request.headers.get("X-Request-Id") or new_request_id()
    log_event("health_check", started_at_ms=started_at_ms, request_id=request_id, deep=deep)
    payload = {
        "ok": True,
        "status": "ok",
        "model_loaded": ENGINE.is_ready(),
        "api_ready": True,
        "deep": deep,
        "request_id": request_id,
        "build_version": BUILD_VERSION,
        "model": MODEL_NAME,
        "gpu": GPU_TYPE,
        "ts": utc_now_iso(),
    }
    if deep:
        try:
            engine = await asyncio.wrap_future(submit_warmup())
        except Exception as exc:
            # Modelo nao carregou: responde degradado em vez de 500.
            engine = {"ok": False, "status": "not_ready", "error": str(exc)[:1000]}
        payload["engine"] = engine
        payload["model_loaded"] = bool(engine.get("status") == "ready")
        payload["status"] = "ok" if payload["model_loaded"] else "degraded"
    return payload


@web.post("/warmup")
async def warmup(request: Request):
    require_auth(request)
    started_at_ms = monotonic_ms()
    request_id = request.headers.get("X-Request-Id") or new_request_id()
    log_event("warmup_request_received", started_at_ms=started_at_ms, request_id=request_id)
    try:
        result = await asyncio.wrap_future(submit_warmup())
    except Exception as exc:
        log_event("warmup_request_failed", started_at_ms=started_at_ms, request_id=request_id, error=str(exc)[:2000])
        raise HTTPException(status_code=503, detail={"error": "model_not_loaded", "details": str(exc)[:1000]})
    result["request_id"] = request_id
    result["elapsed_sec_total"] = round((monotonic_ms() - started_at_ms) / 1000, 3)
    log_event("warmup_request_done", started_at_ms=started_at_ms, request_id=request_id, result=result)
    return result


@web.post("/transcriptions/enqueue")
async def enqueue_transcription(request: Request):
    require_auth(request)
    started_at_ms = monotonic_ms()
    header_request_id = request.headers.get("X-Request-Id")
    trace_context = coerce_trace_context(request.headers.get("X-Trace-Context"))
    job_id = new_job_id()

    log_event("enqueue_received", started_at_ms=started_at_ms, job_id=job_id, request_id=header_request_id, trace=trace_context)

    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        input_info, options, extra = await ingest_request_media(request, job_id, header_request_id or new_request_id(), started_at_ms)

        # Idempotency key: prefer header X-Request-Id, fall back to form/body request_id.
        request_id = header_request_id or extra.get("form_request_id") or new_request_id()

        existing_job_id = lookup_idempotent_job(request_id)
        if existing_job_id:
            existing_job = read_json(job_json_path(existing_job_id))
            log_event(
                "enqueue_idempotent_hit",
                started_at_ms=started_at_ms,
                job_id=existing_job_id,
                request_id=request_id,
            )
            return JSONResponse(
                status_code=200,
                content={
                    "ok": True,
                    "status": existing_job.get("status", "queued"),
                    "job_id": existing_job_id,
                    "request_id": request_id,
                    "idempotent_reuse": True,
                    "build_version": BUILD_VERSION,
                    "status_url": f"/transcriptions/{existing_job_id}",
                    "elapsed_sec": round((monotonic_ms() - started_at_ms) / 1000, 3),
                },
            )

        job = {
            "ok": True,
            "status": "queued",
            "stage": "queued",
            "result_available": False,
            "job_id": job_id,
            "request_id": request_id,
            "trace": trace_context,
            "build_version": BUILD_VERSION,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "input": input_info,
            "options": options,
            "callback": redacted_callback_info(options.get("callback_url")),
        }
        atomic_write_json(job_json_path(job_id), job)
        store_idempotent_job(request_id, job_id)
        commit_volume_safely()

        log_event(
            "enqueue_validated",
            started_at_ms=started_at_ms,
            job_id=job_id,
            request_id=request_id,
            input=input_info,
            callback=job["callback"],
            options={k: v for k, v in options.items() if k not in {"initial_prompt"}},
            initial_prompt_chars=len(options.get("initial_prompt") or ""),
        )

        # Fire-and-forget processing na fila da GPU. O campo worker_ref
        # identifica onde o job foi enfileirado; e gravado ANTES de enfileirar
        # porque o worker pode pegar o job imediatamente.
        job["worker_ref"] = f"local-queue:{job_id}"
        atomic_write_json(job_json_path(job_id), job)
        commit_volume_safely()
        submit_job(job_id)

        log_event(
            "enqueue_done",
            started_at_ms=started_at_ms,
            job_id=job_id,
            request_id=request_id,
            worker_ref=job.get("worker_ref"),
        )

        return JSONResponse(
            status_code=202,
            content={
                "ok": True,
                "status": "queued",
                "job_id": job_id,
                "request_id": request_id,
                "build_version": BUILD_VERSION,
                "created_at": job["created_at"],
                "input": input_info,
                "callback": job["callback"],
                "status_url": f"/transcriptions/{job_id}",
                "elapsed_sec": round((monotonic_ms() - started_at_ms) / 1000, 3),
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        log_event("enqueue_failed", started_at_ms=started_at_ms, job_id=job_id, request_id=header_request_id, error=str(exc)[:4000])
        raise HTTPException(status_code=500, detail={"error": "enqueue_failed", "details": str(exc)[:1000]})


def _extract_media_url_from_payload(body: dict[str, Any]) -> str | None:
    """Best-effort extraction of a media URL from a generic provider payload.

    Supports flat keys (mediaUrl/media_url/url/audioUrl) and a few common nested
    shapes (e.g. n8n/WhatsApp-style: message.media.url, data.media_url). Returns
    the first non-empty string found, or None.
    """
    flat_keys = ("mediaUrl", "media_url", "url", "audioUrl", "audio_url", "fileUrl", "file_url")
    for key in flat_keys:
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    nested_paths = [
        ("message", "media", "url"),
        ("message", "mediaUrl"),
        ("data", "media_url"),
        ("data", "mediaUrl"),
        ("data", "url"),
        ("payload", "mediaUrl"),
        ("payload", "media_url"),
    ]
    for path in nested_paths:
        node: Any = body
        for part in path:
            if isinstance(node, dict):
                node = node.get(part)
            else:
                node = None
                break
        if isinstance(node, str) and node.strip():
            return node.strip()
    return None


@web.post("/webhooks/ingest")
@web.post("/webhook")
async def webhook_ingest(request: Request):
    """Provider-agnostic webhook entrypoint (behind ENABLE_WEBHOOK_INGEST).

    Accepts a JSON payload from an external system (e.g. n8n / WhatsApp bridge),
    extracts a media URL and optional callback_url/request_id, then creates the
    same kind of async job as /transcriptions/enqueue. Returns 202 immediately.

    Idempotency: uses X-Request-Id header, then body request_id/eventId/id.
    """
    require_auth(request)
    if not ENABLE_WEBHOOK_INGEST:
        raise HTTPException(status_code=404, detail="webhook_ingest_disabled")

    started_at_ms = monotonic_ms()
    header_request_id = request.headers.get("X-Request-Id")
    trace_context = coerce_trace_context(request.headers.get("X-Trace-Context"))
    job_id = new_job_id()

    log_event("webhook_ingest_received", started_at_ms=started_at_ms, job_id=job_id, request_id=header_request_id)

    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="json body must be an object")

        media_url = _extract_media_url_from_payload(body)
        if not media_url:
            raise HTTPException(status_code=400, detail="no media url found in payload")

        # Derive idempotency key from header or common provider identifiers.
        body_request_id = (
            str(body.get("request_id") or body.get("eventId") or body.get("id") or "").strip()
            or None
        )
        request_id = header_request_id or body_request_id or new_request_id()

        JOBS_DIR.mkdir(parents=True, exist_ok=True)

        existing_job_id = lookup_idempotent_job(request_id)
        if existing_job_id:
            existing_job = read_json(job_json_path(existing_job_id))
            log_event("webhook_ingest_idempotent_hit", started_at_ms=started_at_ms, job_id=existing_job_id, request_id=request_id)
            return JSONResponse(
                status_code=200,
                content={
                    "ok": True,
                    "status": existing_job.get("status", "queued"),
                    "job_id": existing_job_id,
                    "request_id": request_id,
                    "idempotent_reuse": True,
                    "build_version": BUILD_VERSION,
                    "status_url": f"/transcriptions/{existing_job_id}",
                    "elapsed_sec": round((monotonic_ms() - started_at_ms) / 1000, 3),
                },
            )

        options = build_options(body)
        target = job_dir(job_id) / "input_media"
        input_info = download_media_url(media_url, target, MAX_MEDIA_URL_BYTES, request_id, started_at_ms)
        input_info["source"] = "webhook_ingest"

        job = {
            "ok": True,
            "status": "queued",
            "stage": "queued",
            "result_available": False,
            "job_id": job_id,
            "request_id": request_id,
            "trace": trace_context,
            "build_version": BUILD_VERSION,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "input": input_info,
            "options": options,
            "callback": redacted_callback_info(options.get("callback_url")),
        }
        atomic_write_json(job_json_path(job_id), job)
        store_idempotent_job(request_id, job_id)
        commit_volume_safely()

        job["worker_ref"] = f"local-queue:{job_id}"
        atomic_write_json(job_json_path(job_id), job)
        commit_volume_safely()
        submit_job(job_id)

        log_event("webhook_ingest_done", started_at_ms=started_at_ms, job_id=job_id, request_id=request_id, worker_ref=job.get("worker_ref"))

        return JSONResponse(
            status_code=202,
            content={
                "ok": True,
                "status": "queued",
                "job_id": job_id,
                "request_id": request_id,
                "build_version": BUILD_VERSION,
                "created_at": job["created_at"],
                "input": input_info,
                "callback": job["callback"],
                "status_url": f"/transcriptions/{job_id}",
                "elapsed_sec": round((monotonic_ms() - started_at_ms) / 1000, 3),
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        log_event("webhook_ingest_failed", started_at_ms=started_at_ms, job_id=job_id, request_id=header_request_id, error=str(exc)[:4000])
        raise HTTPException(status_code=500, detail={"error": "webhook_ingest_failed", "details": str(exc)[:1000]})


@web.get("/transcriptions/{job_id}")
async def get_transcription_status(job_id: str, request: Request, include_result: bool = True):
    require_auth(request)
    started_at_ms = monotonic_ms()
    reload_volume_safely()
    path = job_json_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="job_not_found")
    job = read_json(path)

    status = str(job.get("status") or "").lower()
    result_exists = result_json_path(job_id).exists()
    is_completed = status in COMPLETED_STATUSES
    # result_available is authoritative for the client (does not depend on status string).
    result_available = bool(is_completed and result_exists)
    job["result_available"] = result_available

    if include_result and result_exists:
        job["result_payload"] = read_json(result_json_path(job_id))

    log_event(
        "status_read",
        started_at_ms=started_at_ms,
        job_id=job_id,
        request_id=job.get("request_id"),
        status=job.get("status"),
        stage=job.get("stage"),
        result_available=result_available,
        include_result=include_result,
    )
    return job


@web.get("/transcriptions/{job_id}/result")
async def get_transcription_result(job_id: str, request: Request):
    """Devolve so o result_payload de um job concluido.

    O GET /transcriptions/{job_id} ja embute o payload; este endpoint atende
    clientes (stt-cli, Atlas) que buscam o resultado separadamente quando o
    payload nao vem no status.
    """
    require_auth(request)
    reload_volume_safely()
    path = result_json_path(job_id)
    if not job_json_path(job_id).exists():
        raise HTTPException(status_code=404, detail="job_not_found")
    if not path.exists():
        raise HTTPException(status_code=409, detail="result_not_ready")
    return read_json(path)


@web.post("/transcribe")
@web.post("/transcribe-sync-debug")
async def transcribe_sync_debug(request: Request):
    """Synchronous debug endpoint. Prefer /transcriptions/enqueue for the regular flow."""
    require_auth(request)
    started_at_ms = monotonic_ms()
    request_id = request.headers.get("X-Request-Id") or new_request_id()
    trace_context = coerce_trace_context(request.headers.get("X-Trace-Context"))
    job_id = new_job_id()

    log_event("sync_transcribe_received", started_at_ms=started_at_ms, job_id=job_id, request_id=request_id)
    input_info, options, _extra = await ingest_request_media(request, job_id, request_id, started_at_ms)
    job = {
        "ok": True,
        "status": "processing",
        "stage": "sync_debug",
        "result_available": False,
        "job_id": job_id,
        "request_id": request_id,
        "trace": trace_context,
        "build_version": BUILD_VERSION,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "input": input_info,
        "options": options,
        "callback": redacted_callback_info(options.get("callback_url")),
    }
    atomic_write_json(job_json_path(job_id), job)
    commit_volume_safely()
    result = await asyncio.wrap_future(submit_job(job_id))
    result["sync_endpoint_elapsed_sec"] = round((monotonic_ms() - started_at_ms) / 1000, 3)
    return result


# =============================================================================
# Retention / cleanup (POST /maintenance/cleanup)
# =============================================================================


def cleanup_old_jobs() -> dict[str, Any]:
    """Delete job directories older than JOB_RETENTION_HOURS.

    Chamado por POST /maintenance/cleanup (agendador externo). Removes
    both /data/jobs/<job_id>/ and orphaned /data/idempotency/ entries whose job
    directory no longer exists. Age is derived from job.json.created_at when
    available, falling back to the directory mtime.
    """
    started_at_ms = monotonic_ms()
    reload_volume_safely()

    cutoff = datetime.now(timezone.utc).timestamp() - (JOB_RETENTION_HOURS * 3600)
    removed_jobs: list[str] = []
    kept = 0
    errors = 0

    if JOBS_DIR.exists():
        for entry in JOBS_DIR.iterdir():
            if not entry.is_dir():
                continue
            try:
                created_ts: float | None = None
                jpath = entry / "job.json"
                if jpath.exists():
                    try:
                        created_at = read_json(jpath).get("created_at")
                        if created_at:
                            created_ts = datetime.fromisoformat(created_at).timestamp()
                    except Exception:
                        created_ts = None
                if created_ts is None:
                    created_ts = entry.stat().st_mtime

                if created_ts < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed_jobs.append(entry.name)
                else:
                    kept += 1
            except Exception as exc:
                errors += 1
                log_event("cleanup_job_error", job_id=entry.name, error=str(exc)[:1000])

    # Remove idempotency pointers whose job directory is gone.
    removed_idempotency = 0
    if IDEMPOTENCY_DIR.exists():
        for entry in IDEMPOTENCY_DIR.iterdir():
            if not entry.is_file():
                continue
            try:
                target_job = read_json(entry).get("job_id")
                if target_job and not (JOBS_DIR / target_job).exists():
                    entry.unlink()
                    removed_idempotency += 1
            except Exception:
                # Best-effort; a malformed pointer is safe to drop.
                try:
                    entry.unlink()
                    removed_idempotency += 1
                except Exception:
                    pass

    commit_volume_safely()
    summary = {
        "ok": True,
        "retention_hours": JOB_RETENTION_HOURS,
        "removed_jobs_count": len(removed_jobs),
        "kept_jobs_count": kept,
        "removed_idempotency_count": removed_idempotency,
        "errors": errors,
    }
    log_event("cleanup_done", started_at_ms=started_at_ms, **summary)
    return summary


@web.post("/maintenance/cleanup")
def maintenance_cleanup(request: Request):
    """Dispara a limpeza de retencao (chamado por um agendador externo).

    Exige TRANSCRIBE_API_TOKEN configurado: sem token a rota fica fechada, para
    ninguem apagar jobs anonimamente. Endpoint sincrono (def) para o FastAPI
    rodar em threadpool sem travar o event loop.
    """
    if not os.getenv(API_TOKEN_ENV, "").strip():
        raise HTTPException(status_code=403, detail="maintenance_requires_api_token")
    require_auth(request)
    with _IN_FLIGHT_LOCK:
        in_flight = set(_IN_FLIGHT)
    if in_flight:
        log_event("cleanup_with_jobs_in_flight", in_flight_count=len(in_flight))
    return cleanup_old_jobs()


# =============================================================================
# Entrypoint (uvicorn direto)
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # workers=1 e obrigatorio: o modelo e a fila sao singletons do processo.
    uvicorn.run(
        web,
        host=os.getenv("HOST", "0.0.0.0"),
        port=parse_int(os.getenv("PORT"), 8000),
        workers=1,
        timeout_keep_alive=75,
    )
