"""Cliente de teste do servico de transcricao.

Uso:
    uv run test_client.py --base-url URL --file audio.ogg --poll [--token T]
    uv run test_client.py --base-url URL --warmup [--token T]

Com --file: chama /warmup, envia o arquivo em /transcriptions/enqueue e, com
--poll, consulta /transcriptions/{job_id} ate o job terminar. Imprime o texto e
os tempos (ate o warmup responder, ate o enqueue, ate o resultado).
Com --warmup: so chama /warmup e mede o tempo.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
from pathlib import Path
from typing import Any

import requests

COMPLETED_STATUSES = {"completed", "done", "success", "succeeded", "finished"}
FAILED_STATUSES = {"failed", "error"}
# Enquanto a replica sobe (scale-from-zero) o ingress pode devolver estes codigos.
RETRYABLE_STATUS = {502, 503, 504}


def auth_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def fail(message: str, response: requests.Response | None = None) -> None:
    print(f"ERRO: {message}", file=sys.stderr)
    if response is not None:
        print(f"  HTTP {response.status_code}: {response.text[:2000]}", file=sys.stderr)
    sys.exit(1)


def call_warmup(base_url: str, token: str | None, timeout_sec: float) -> tuple[dict[str, Any], float]:
    """POST /warmup, repetindo ate responder 200 ou estourar timeout_sec.

    O primeiro warmup depois de escalar a zero inclui subir a replica e carregar
    o modelo, e o ingress corta requests longos; por isso a repeticao.
    """
    started = time.monotonic()
    deadline = started + timeout_sec
    attempt = 0
    while True:
        attempt += 1
        last_error = ""
        try:
            response = requests.post(f"{base_url}/warmup", headers=auth_headers(token), timeout=(10, 300))
            if response.status_code == 200:
                return response.json(), time.monotonic() - started
            if response.status_code not in RETRYABLE_STATUS:
                fail("warmup falhou", response)
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)[:200]
        if time.monotonic() >= deadline:
            fail(f"warmup nao respondeu em {timeout_sec:.0f}s (ultimo erro: {last_error})")
        print(f"  warmup tentativa {attempt}: {last_error}; tentando de novo...", file=sys.stderr)
        time.sleep(3)


def enqueue_file(base_url: str, token: str | None, path: Path, language: str | None) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = {"language": language} if language else {}
    with path.open("rb") as f:
        response = requests.post(
            f"{base_url}/transcriptions/enqueue",
            headers=auth_headers(token),
            files={"file": (path.name, f, content_type)},
            data=data,
            timeout=(10, 600),
        )
    if response.status_code not in (200, 202):
        fail("enqueue falhou", response)
    return response.json(), time.monotonic() - started


def poll_job(
    base_url: str, token: str | None, job_id: str, interval_sec: float, timeout_sec: float
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    deadline = started + timeout_sec
    last_seen = ""
    while True:
        try:
            response = requests.get(
                f"{base_url}/transcriptions/{job_id}", headers=auth_headers(token), timeout=(10, 60)
            )
        except requests.RequestException as exc:
            response = None
            print(f"  polling: {str(exc)[:200]}; tentando de novo...", file=sys.stderr)
        if response is not None:
            if response.status_code == 200:
                job = response.json()
                status = str(job.get("status") or "").lower()
                seen = f"{status}/{job.get('stage')}"
                if seen != last_seen:
                    print(f"  [{time.monotonic() - started:6.1f}s] status={status} stage={job.get('stage')}", file=sys.stderr)
                    last_seen = seen
                if status in COMPLETED_STATUSES or status in FAILED_STATUSES:
                    return job, time.monotonic() - started
            elif response.status_code not in RETRYABLE_STATUS:
                fail("consulta de status falhou", response)
        if time.monotonic() >= deadline:
            fail(f"job {job_id} nao terminou em {timeout_sec:.0f}s")
        time.sleep(interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cliente de teste do APO Whisper")
    parser.add_argument("--base-url", required=True, help="URL base do servico (ex.: http://localhost:8000)")
    parser.add_argument("--token", default=None, help="valor de TRANSCRIBE_API_TOKEN, se a API exigir")
    parser.add_argument("--file", type=Path, default=None, help="arquivo de audio/video a transcrever")
    parser.add_argument("--poll", action="store_true", help="aguarda o job terminar e imprime o texto")
    parser.add_argument("--warmup", action="store_true", help="so chama /warmup e mede o tempo")
    parser.add_argument("--language", default=None, help="idioma do audio (padrao do servico: pt)")
    parser.add_argument("--interval", type=float, default=2.0, help="intervalo do polling em segundos")
    parser.add_argument("--timeout", type=float, default=1800.0, help="tempo maximo de espera em segundos")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    if not args.warmup and args.file is None:
        parser.error("informe --file ARQUIVO ou --warmup")
    if args.file is not None and not args.file.is_file():
        parser.error(f"arquivo nao encontrado: {args.file}")

    total_started = time.monotonic()
    warmup, warmup_sec = call_warmup(base_url, args.token, args.timeout)
    print(
        f"warmup: status={warmup.get('status')} model={warmup.get('model')} gpu={warmup.get('gpu')} "
        f"compute_type={warmup.get('compute_type')} build={warmup.get('build_version')}",
        file=sys.stderr,
    )
    if args.warmup:
        print(json.dumps(warmup, ensure_ascii=False, indent=2))
        print(f"\nTempo ate o warmup responder: {warmup_sec:.2f}s")
        return

    enqueued, enqueue_sec = enqueue_file(base_url, args.token, args.file, args.language)
    job_id = enqueued["job_id"]
    print(f"enqueue: job_id={job_id} status={enqueued.get('status')}", file=sys.stderr)

    result_sec: float | None = None
    failed = False
    if args.poll:
        job, result_sec = poll_job(base_url, args.token, job_id, args.interval, args.timeout)
        payload = job.get("result_payload") or {}
        if str(job.get("status") or "").lower() in FAILED_STATUSES:
            failed = True
            print(f"ERRO: job falhou: {payload.get('details') or job.get('error')}", file=sys.stderr)
        else:
            metrics = payload.get("metrics") or {}
            print("\n--- texto ---")
            print((payload.get("result") or {}).get("text", ""))
            print("-------------")
            print(
                f"audio: {metrics.get('media_duration_sec_sum')}s | processamento no servidor: "
                f"{metrics.get('elapsed_sec')}s | velocidade: {metrics.get('overall_speed_x')}x"
            )
    else:
        print(f"job_id: {job_id} (use --poll para aguardar o resultado)")

    print("\nTempos:")
    print(f"  ate o warmup responder: {warmup_sec:.2f}s")
    print(f"  ate o enqueue:          {enqueue_sec:.2f}s")
    if result_sec is not None:
        print(f"  ate o resultado:        {result_sec:.2f}s")
    print(f"  total:                  {time.monotonic() - total_started:.2f}s")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
