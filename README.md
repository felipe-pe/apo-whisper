# apo-whisper

Serviço HTTP de transcrição de áudio e vídeo com [faster-whisper](https://github.com/SYSTRAN/faster-whisper) em GPU, feito para rodar no **Azure Container Apps** com GPU serverless (perfil `gpu-t4`, NC8as T4).

## O que o serviço faz

- Recebe mídia por upload (multipart), por URL (`mediaUrl`) ou no corpo cru do request.
- Extrai e normaliza o áudio com ffmpeg (mono 16 kHz, highpass e normalização de loudness conforme o perfil de mídia: `whatsapp`, `meeting`, `raw` ou `auto`).
- Transcreve com o Whisper `large-v3` (VAD, filtro de alucinação, uma transcrição por faixa de áudio).
- Processa um job por vez: fila em memória com um worker na GPU. O estado de cada job fica em disco (`job.json` e `result.json` em `/data/jobs/<job_id>/`), com idempotência por `X-Request-Id`.
- Opcionalmente envia o resultado para um `callback_url`, com retry e assinatura HMAC.
- No startup, recoloca na fila os jobs que estavam `queued`/`processing` em disco (até `RECOVER_MAX_AGE_HOURS`).

O modelo carrega em background, então a porta abre na hora; `/warmup` e os jobs aguardam o modelo na fila.

## Rotas

Se `TRANSCRIBE_API_TOKEN` estiver definido, todas exigem `Authorization: Bearer <token>`.

| Rota | Descrição |
|---|---|
| `GET /health` | Estado da API e `model_loaded`. Com `?deep=true` roda também um warmup. |
| `POST /warmup` | Aguarda o modelo carregar e roda uma inferência sintética. |
| `POST /transcriptions/enqueue` | Cria um job assíncrono e responde `202` com `job_id`. Aceita multipart (`file` + opções), JSON (`mediaUrl` + opções) ou corpo cru. |
| `GET /transcriptions/{job_id}` | Status do job. Quando termina, traz o resultado em `result_payload` (`?include_result=false` omite). |
| `GET /transcriptions/{job_id}/result` | Só o resultado (`409` enquanto não estiver pronto). |
| `POST /transcribe` | Transcrição síncrona, para depuração. Alias: `/transcribe-sync-debug`. |
| `POST /webhooks/ingest` | Cria um job a partir de um payload JSON com URL de mídia. Exige `ENABLE_WEBHOOK_INGEST=true`. Alias: `/webhook`. |
| `POST /maintenance/cleanup` | Apaga jobs mais antigos que `JOB_RETENTION_HOURS`. Só funciona com `TRANSCRIBE_API_TOKEN` configurado. |

Opções por request (campos do form ou do JSON): `language`, `media_profile`, `audio_preprocess`, `initial_prompt`, `hotwords`, `callback_url`, `word_timestamps`, `split_tracks`, `vad_filter`, `beam_size`, `request_id`, entre outras (ver `build_options` em `app.py`).

O texto final fica em `result.text`; `result.tracks` traz segmentos e sinais de qualidade por faixa. O `job.json` inclui o campo `worker_ref` (`local-queue:<job_id>`).

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `TRANSCRIBE_API_TOKEN` | vazio | Token Bearer da API. Vazio = API sem autenticação. |
| `WHISPER_MODEL_NAME` | `large-v3` | Modelo do faster-whisper. |
| `WHISPER_COMPUTE_TYPE` | `float16` | Tipo de computação na GPU. |
| `WHISPER_COMPUTE_TYPE_FALLBACK` | `int8_float16` | Usado se o tipo principal falhar. |
| `MODEL_DIR` | `/models` | Pasta do modelo (dentro da imagem). |
| `DATA_DIR` | `/data` | Pasta de dados (Azure Files). |
| `JOBS_DIR` | `$DATA_DIR/jobs` | Pasta dos jobs. |
| `IDEMPOTENCY_DIR` | `$DATA_DIR/idempotency` | Ponteiros de idempotência. |
| `WHISPER_INITIAL_PROMPT` | vazio | `initial_prompt` padrão quando o request não envia. |
| `WHISPER_HOTWORDS` | vazio | `hotwords` padrão quando o request não envia. |
| `DEFAULT_CALLBACK_URL` | vazio | `callback_url` padrão. |
| `CALLBACK_SIGNING_SECRET` | vazio | Se definido, o callback leva `X-Signature` (HMAC-SHA256 do corpo). |
| `CALLBACK_MAX_RETRIES` / `CALLBACK_BACKOFF_BASE_SEC` | `3` / `2.0` | Retry do callback. |
| `ENABLE_WEBHOOK_INGEST` | `false` | Habilita `/webhooks/ingest`. |
| `DEFAULT_MEDIA_PROFILE` | `auto` | `auto`, `whatsapp`, `meeting` ou `raw`. |
| `AUTO_PROFILE_DURATION_SPLIT_SEC` | `180` | No perfil `auto`, mídia acima disso é tratada como `meeting`. |
| `AUDIO_PREPROCESS_DEFAULT` | `light` | `none`, `light` ou `aggressive`. |
| `WHISPER_BEST_OF`, `WHISPER_COMPRESSION_RATIO_THRESHOLD`, `WHISPER_LOG_PROB_THRESHOLD`, `WHISPER_NO_SPEECH_THRESHOLD`, `WHISPER_HALLUCINATION_SILENCE_THRESHOLD`, `WHISPER_REPETITION_PENALTY`, `WHISPER_NO_REPEAT_NGRAM_SIZE` | ver `app.py` | Ajustes de decodificação. |
| `JOB_RETENTION_HOURS` | `168` | Retenção usada por `/maintenance/cleanup`. |
| `DELETE_INPUT_AFTER_SUCCESS` | `true` | Apaga a mídia de entrada depois do sucesso. |
| `RECOVER_PENDING_JOBS` / `RECOVER_MAX_AGE_HOURS` | `true` / `24` | Recuperação de jobs no startup. |
| `GPU_TYPE`, `GPU_HOURLY_USD`, `EMIT_BILLING_EVENTS`, `ESTIMATE_GPU_COST` | `T4`, `1.10`, `true`, `true` | Informativos, para os eventos de custo nos logs. |
| `GPU_SCALEDOWN_WINDOW_SEC`, `GPU_MIN_CONTAINERS`, `GPU_MAX_CONTAINERS` | `300`, `0`, `1` | Informativos; o `deploy.sh` preenche com a escala configurada. |
| `BUILD_VERSION` | `azure-aca-whisper-t4-...` | Identificador devolvido nas respostas. |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Usados ao rodar `app.py` diretamente. |

## Instalação

Requer [uv](https://docs.astral.sh/uv/) e, para transcrever, `ffmpeg`/`ffprobe` no `PATH`.

```bash
uv sync
```

## Teste local

Precisa de uma GPU NVIDIA com CUDA 12 e cuDNN 9 (o serviço carrega o modelo com `device=cuda`).

```bash
MODEL_DIR=./models DATA_DIR=./data uv run uvicorn app:web --port 8000 --workers 1
```

Fora da imagem não existe `MODEL_DIR/model.bin`, então o modelo (~3 GB) é baixado para `MODEL_DIR` na primeira carga. Use sempre `--workers 1`: o modelo e a fila são únicos por processo.

Em outro terminal:

```bash
uv run test_client.py --base-url http://localhost:8000 --file audio.ogg --poll
```

## Deploy

```bash
export TRANSCRIBE_API_TOKEN='...'   # sem token a API fica aberta
./deploy.sh
```

O script pode ser executado de novo para atualizar. Ele:

1. Faz `az acr build` da imagem no registry `acrapotranscribe` (com o modelo dentro).
2. Garante a storage account, o file share `whisper-data` e o registro dele no ambiente `cae-apo-transcribe`.
3. Garante a identidade `id-apo-whisper` com permissão AcrPull.
4. Cria ou atualiza o Container App `ca-apo-whisper`: perfil `gpu-t4`, 0 a 1 réplica, ingress externo na porta 8000 e `whisper-data` montado em `/data`.

As variáveis no topo do script podem ser sobrescritas pelo ambiente. As opcionais do app (`ENABLE_WEBHOOK_INGEST`, `WHISPER_HOTWORDS` etc.) são repassadas ao container quando estão definidas no shell.

Pontos de operação:

- **Escala a zero:** o Container App escala por requests HTTP. Um job longo sem nenhum request chegando pode ser interrompido depois do `cooldownPeriod` (`COOLDOWN_SEC`, padrão 300 s) e recomeça do zero quando a réplica volta. Clientes que fazem polling mantêm a réplica viva; se o fluxo usar só callback, aumente `COOLDOWN_SEC`.
- **Requests síncronos:** o ingress corta requests em ~240 s. Para áudios longos, use `/transcriptions/enqueue`, não `/transcribe`.
- **Logs:** `az containerapp logs show -n ca-apo-whisper -g rg-atlas-eus --follow` (um evento JSON por linha).
- **Retenção:** agende diariamente

  ```bash
  curl -X POST "https://$URL/maintenance/cleanup" -H "Authorization: Bearer $TRANSCRIBE_API_TOKEN"
  ```

## Teste com test_client.py

```bash
URL=$(az containerapp show -n ca-apo-whisper -g rg-atlas-eus --query properties.configuration.ingress.fqdn -o tsv)

# só o warmup: mede o tempo até a réplica subir e o modelo carregar
uv run test_client.py --base-url "https://$URL" --token "$TRANSCRIBE_API_TOKEN" --warmup

# transcrição completa
uv run test_client.py --base-url "https://$URL" --token "$TRANSCRIBE_API_TOKEN" --file audio.ogg --poll
```

Com `--file`, o cliente chama `/warmup`, envia o arquivo em `/transcriptions/enqueue` e, com `--poll`, consulta `/transcriptions/{job_id}` até o job terminar. No fim imprime o texto e os tempos: até o warmup responder, até o enqueue e até o resultado. Outras opções: `--language`, `--interval`, `--timeout`.

## Modelo dentro da imagem

O `Dockerfile` baixa o modelo para `/models` durante o build, e o serviço carrega direto de `MODEL_DIR` quando encontra `MODEL_DIR/model.bin`, sem baixar nada. No cold start, puxar a imagem do ACR na mesma região é bem mais rápido do que ler ~3 GB de um file share por SMB. Por isso só `/data` (jobs) fica no Azure Files.

Para trocar de modelo, defina `WHISPER_MODEL_NAME` antes de rodar o `./deploy.sh`: o valor vai para o build da imagem e para o container. Se `MODEL_DIR/model.bin` não existir, o serviço baixa o modelo para `MODEL_DIR` na primeira carga.
