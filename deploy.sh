#!/usr/bin/env bash
# Deploy do servico de transcricao (faster-whisper) no Azure Container Apps com GPU T4.
#
# O que faz (idempotente, pode rodar de novo para atualizar):
#   1. az acr build da imagem (com o modelo dentro) no registry acrapotranscribe
#   2. garante storage account + file share de dados e registra no ambiente
#   3. garante uma identidade gerenciada com AcrPull no registry (admin do ACR e desligado)
#   4. cria ou atualiza o Container App ca-apo-whisper (perfil gpu-t4, 0..1 replicas,
#      ingress externo na 8000, Azure Files em /data, env vars)
#
# NAO toca no ca-apo-transcribe (outro servico no mesmo ambiente).
#
# Uso:
#   export TRANSCRIBE_API_TOKEN=...      # recomendado (sem ele a API fica aberta)
#   ./deploy.sh
#
# Qualquer variavel abaixo pode ser sobrescrita pelo ambiente.
set -euo pipefail

RG="${RG:-rg-atlas-eus}"
ACR_NAME="${ACR_NAME:-acrapotranscribe}"
ENV_NAME="${ENV_NAME:-cae-apo-transcribe}"
APP_NAME="${APP_NAME:-ca-apo-whisper}"
WORKLOAD_PROFILE="${WORKLOAD_PROFILE:-gpu-t4}"
IMAGE_REPO="${IMAGE_REPO:-apo-whisper}"
IMAGE_TAG="${IMAGE_TAG:-$(date -u +%Y%m%d%H%M%S)}"
IDENTITY_NAME="${IDENTITY_NAME:-id-apo-whisper}"

# Storage (Azure Files). O nome da conta precisa ser unico no Azure; o padrao
# deriva um sufixo estavel da subscription para o script ser re-executavel.
DATA_SHARE="${DATA_SHARE:-whisper-data}"
ENV_STORAGE_DATA="${ENV_STORAGE_DATA:-whisperdata}"
SHARE_QUOTA_GB="${SHARE_QUOTA_GB:-100}"

# Escala: 0..1 replica. COOLDOWN_SEC e quanto tempo sem requests ate escalar a zero.
MIN_REPLICAS="${MIN_REPLICAS:-0}"
MAX_REPLICAS="${MAX_REPLICAS:-1}"
COOLDOWN_SEC="${COOLDOWN_SEC:-300}"
CPU="${CPU:-8}"
MEMORY="${MEMORY:-56Gi}"

# Config da aplicacao.
WHISPER_MODEL_NAME="${WHISPER_MODEL_NAME:-large-v3}"
WHISPER_COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE:-float16}"
TRANSCRIBE_API_TOKEN="${TRANSCRIBE_API_TOKEN:-}"
CALLBACK_SIGNING_SECRET="${CALLBACK_SIGNING_SECRET:-}"

# Variaveis opcionais repassadas ao container se estiverem definidas no shell.
PASSTHROUGH_VARS=(
  DEFAULT_CALLBACK_URL WHISPER_INITIAL_PROMPT WHISPER_HOTWORDS AUDIO_PREPROCESS_DEFAULT
  DEFAULT_MEDIA_PROFILE AUTO_PROFILE_DURATION_SPLIT_SEC
  WHISPER_BEST_OF WHISPER_COMPRESSION_RATIO_THRESHOLD WHISPER_LOG_PROB_THRESHOLD
  WHISPER_NO_SPEECH_THRESHOLD WHISPER_HALLUCINATION_SILENCE_THRESHOLD
  WHISPER_REPETITION_PENALTY WHISPER_NO_REPEAT_NGRAM_SIZE WHISPER_COMPUTE_TYPE_FALLBACK
  ENABLE_WEBHOOK_INGEST CALLBACK_MAX_RETRIES CALLBACK_BACKOFF_BASE_SEC
  JOB_RETENTION_HOURS DELETE_INPUT_AFTER_SUCCESS RECOVER_PENDING_JOBS RECOVER_MAX_AGE_HOURS
  EMIT_BILLING_EVENTS ESTIMATE_GPU_COST GPU_HOURLY_USD
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log() { printf '\n[deploy] %s\n' "$*"; }
die() { printf '\n[deploy] ERRO: %s\n' "$*" >&2; exit 1; }
# Escapa string para YAML em aspas simples.
yq() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/''/g")"; }

[[ "$APP_NAME" == "ca-apo-transcribe" ]] && die "APP_NAME nao pode ser ca-apo-transcribe (outro servico)."

log "Garantindo extensao containerapp do az CLI"
az extension add --name containerapp --upgrade --only-show-errors -y >/dev/null

SUBSCRIPTION_ID="$(az account show --query id -o tsv)"
if [[ -z "${STORAGE_ACCOUNT:-}" ]]; then
  STORAGE_ACCOUNT="stapowhisper$(printf '%s' "$SUBSCRIPTION_ID" | cksum | cut -d' ' -f1 | cut -c1-8)"
fi

log "Validando ambiente $ENV_NAME e perfil $WORKLOAD_PROFILE"
ENV_ID="$(az containerapp env show -n "$ENV_NAME" -g "$RG" --query id -o tsv)"
LOCATION="$(az containerapp env show -n "$ENV_NAME" -g "$RG" --query location -o tsv)"
az containerapp env show -n "$ENV_NAME" -g "$RG" \
  --query "properties.workloadProfiles[?name=='$WORKLOAD_PROFILE'].name" -o tsv | grep -qx "$WORKLOAD_PROFILE" \
  || die "workload profile '$WORKLOAD_PROFILE' nao existe em $ENV_NAME."

# -----------------------------------------------------------------------------
# 1. Build da imagem no ACR (o modelo e baixado para dentro da imagem; build
#    remoto, nao precisa Docker local)
# -----------------------------------------------------------------------------
ACR_SERVER="$(az acr show -n "$ACR_NAME" --query loginServer -o tsv)"
ACR_ID="$(az acr show -n "$ACR_NAME" --query id -o tsv)"
IMAGE="$ACR_SERVER/$IMAGE_REPO:$IMAGE_TAG"
log "az acr build -> $IMAGE"
az acr build --registry "$ACR_NAME" --image "$IMAGE_REPO:$IMAGE_TAG" \
  --build-arg WHISPER_MODEL_NAME="$WHISPER_MODEL_NAME" --file "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# 2. Storage: conta + share de dados + registro no ambiente (cria se faltar)
# -----------------------------------------------------------------------------
if ! az storage account show -n "$STORAGE_ACCOUNT" -g "$RG" -o none 2>/dev/null; then
  log "Criando storage account $STORAGE_ACCOUNT"
  az storage account create -n "$STORAGE_ACCOUNT" -g "$RG" -l "$LOCATION" \
    --sku Standard_LRS --kind StorageV2 --min-tls-version TLS1_2 --allow-blob-public-access false -o none
fi
if ! az storage share-rm show --storage-account "$STORAGE_ACCOUNT" -g "$RG" -n "$DATA_SHARE" -o none 2>/dev/null; then
  log "Criando file share $DATA_SHARE"
  az storage share-rm create --storage-account "$STORAGE_ACCOUNT" -g "$RG" -n "$DATA_SHARE" --quota "$SHARE_QUOTA_GB" -o none
fi
STORAGE_KEY="$(az storage account keys list -n "$STORAGE_ACCOUNT" -g "$RG" --query '[0].value' -o tsv)"
ensure_env_storage() {
  local name="$1" share="$2"
  if ! az containerapp env storage show -n "$ENV_NAME" -g "$RG" --storage-name "$name" -o none 2>/dev/null; then
    log "Registrando storage $name ($share) no ambiente $ENV_NAME"
    az containerapp env storage set -n "$ENV_NAME" -g "$RG" --storage-name "$name" \
      --azure-file-account-name "$STORAGE_ACCOUNT" --azure-file-account-key "$STORAGE_KEY" \
      --azure-file-share-name "$share" --access-mode ReadWrite -o none
  fi
}
ensure_env_storage "$ENV_STORAGE_DATA" "$DATA_SHARE"

# -----------------------------------------------------------------------------
# 3. Identidade para puxar a imagem do ACR (admin user do ACR esta desligado)
# -----------------------------------------------------------------------------
if ! az identity show -n "$IDENTITY_NAME" -g "$RG" -o none 2>/dev/null; then
  log "Criando identidade $IDENTITY_NAME"
  az identity create -n "$IDENTITY_NAME" -g "$RG" -l "$LOCATION" -o none
fi
IDENTITY_ID="$(az identity show -n "$IDENTITY_NAME" -g "$RG" --query id -o tsv)"
IDENTITY_PRINCIPAL="$(az identity show -n "$IDENTITY_NAME" -g "$RG" --query principalId -o tsv)"
if [[ -z "$(az role assignment list --assignee "$IDENTITY_PRINCIPAL" --scope "$ACR_ID" --role AcrPull --query '[0].id' -o tsv)" ]]; then
  log "Concedendo AcrPull para $IDENTITY_NAME (aguardando propagacao)"
  az role assignment create --assignee-object-id "$IDENTITY_PRINCIPAL" --assignee-principal-type ServicePrincipal \
    --role AcrPull --scope "$ACR_ID" -o none
  sleep 45
fi

# -----------------------------------------------------------------------------
# 4. Container App (YAML: unico jeito do az CLI montar volumes Azure Files)
# -----------------------------------------------------------------------------
YAML_FILE="$(mktemp -t ca-apo-whisper.XXXXXX)"
trap 'rm -f "$YAML_FILE"' EXIT

SECRETS_YAML=""
SECRET_ENV_YAML=""
if [[ -n "$TRANSCRIBE_API_TOKEN" ]]; then
  SECRETS_YAML+="      - name: transcribe-api-token
        value: $(yq "$TRANSCRIBE_API_TOKEN")
"
  SECRET_ENV_YAML+="            - name: TRANSCRIBE_API_TOKEN
              secretRef: transcribe-api-token
"
else
  log "AVISO: TRANSCRIBE_API_TOKEN vazio -> API sem autenticacao e /maintenance/cleanup bloqueado."
fi
if [[ -n "$CALLBACK_SIGNING_SECRET" ]]; then
  SECRETS_YAML+="      - name: callback-signing-secret
        value: $(yq "$CALLBACK_SIGNING_SECRET")
"
  SECRET_ENV_YAML+="            - name: CALLBACK_SIGNING_SECRET
              secretRef: callback-signing-secret
"
fi
[[ -z "$SECRETS_YAML" ]] && SECRETS_YAML="      []
"

env_item() { printf '            - name: %s\n              value: %s\n' "$1" "$(yq "$2")"; }
ENV_YAML="$(
  env_item MODEL_DIR /models
  env_item DATA_DIR /data
  env_item JOBS_DIR /data/jobs
  env_item GPU_TYPE T4
  env_item WHISPER_MODEL_NAME "$WHISPER_MODEL_NAME"
  env_item WHISPER_COMPUTE_TYPE "$WHISPER_COMPUTE_TYPE"
  env_item GPU_SCALEDOWN_WINDOW_SEC "$COOLDOWN_SEC"
  env_item GPU_MIN_CONTAINERS "$MIN_REPLICAS"
  env_item GPU_MAX_CONTAINERS "$MAX_REPLICAS"
  for var in "${PASSTHROUGH_VARS[@]}"; do
    if [[ -n "${!var:-}" ]]; then env_item "$var" "${!var}"; fi
  done
)"

cat >"$YAML_FILE" <<EOF
location: $(yq "$LOCATION")
identity:
  type: UserAssigned
  userAssignedIdentities:
    $IDENTITY_ID: {}
properties:
  environmentId: $ENV_ID
  workloadProfileName: $WORKLOAD_PROFILE
  configuration:
    activeRevisionsMode: Single
    ingress:
      external: true
      targetPort: 8000
      transport: auto
      allowInsecure: false
    registries:
      - server: $ACR_SERVER
        identity: $IDENTITY_ID
    secrets:
$SECRETS_YAML  template:
    containers:
      - name: whisper
        image: $IMAGE
        resources:
          cpu: $CPU
          memory: $MEMORY
        env:
$ENV_YAML
$SECRET_ENV_YAML        volumeMounts:
          - volumeName: data
            mountPath: /data
    scale:
      minReplicas: $MIN_REPLICAS
      maxReplicas: $MAX_REPLICAS
      cooldownPeriod: $COOLDOWN_SEC
      rules:
        - name: http-scale
          http:
            metadata:
              concurrentRequests: '20'
    volumes:
      # nobrl: evita erro de lock de byte-range no SMB (so ha 1 replica escrevendo).
      - name: data
        storageType: AzureFile
        storageName: $ENV_STORAGE_DATA
        mountOptions: 'dir_mode=0777,file_mode=0777,mfsymlinks,nobrl'
EOF

if az containerapp show -n "$APP_NAME" -g "$RG" -o none 2>/dev/null; then
  log "Atualizando Container App $APP_NAME"
  az containerapp update -n "$APP_NAME" -g "$RG" --yaml "$YAML_FILE" -o none
else
  log "Criando Container App $APP_NAME"
  az containerapp create -n "$APP_NAME" -g "$RG" --yaml "$YAML_FILE" -o none
fi

FQDN="$(az containerapp show -n "$APP_NAME" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
log "Pronto. Imagem: $IMAGE"
log "URL base: https://$FQDN"
log "Teste: uv run test_client.py --base-url https://$FQDN --file audio.ogg --poll"
