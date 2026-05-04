#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.env.train}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env file not found: $ENV_FILE"
  echo "Usage: ./run_train.sh [path-to-env-file]"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

TRAIN_MODE="${TRAIN_MODE:-single}"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-olmocr/train/configs/v0.4.0/qwen25_vl_olmocrv4_finetuning.yaml}"
TRAIN_PYTHON_BIN="${TRAIN_PYTHON_BIN:-python}"
TRAIN_ALLOC_CONF="${TRAIN_ALLOC_CONF:-expandable_segments:True}"
TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-29500}"

export PYTORCH_ALLOC_CONF="$TRAIN_ALLOC_CONF"

if [[ "$TRAIN_MODE" == "single" ]]; then
  export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES_SINGLE:-0}"
  echo "[run_train] mode=single cuda=${CUDA_VISIBLE_DEVICES} config=${TRAIN_CONFIG_PATH}"
  exec "$TRAIN_PYTHON_BIN" -m olmocr.train.train --config "$TRAIN_CONFIG_PATH"
fi

if [[ "$TRAIN_MODE" == "multi" ]]; then
  export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES_MULTI:-0,1}"
  NPROC="${TRAIN_NPROC_PER_NODE:-}"
  if [[ -z "$NPROC" ]]; then
    NPROC="$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
  fi
  echo "[run_train] mode=multi cuda=${CUDA_VISIBLE_DEVICES} nproc=${NPROC} config=${TRAIN_CONFIG_PATH}"
  exec "$TRAIN_PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node "$NPROC" \
    --master_port "$TRAIN_MASTER_PORT" \
    -m olmocr.train.train --config "$TRAIN_CONFIG_PATH"
fi

echo "Unsupported TRAIN_MODE: $TRAIN_MODE (expected: single or multi)"
exit 1
