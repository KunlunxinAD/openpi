#!/usr/bin/env bash
set -euo pipefail

# Paths and network.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/workspace/models/openpi_data}"
export HF_HOME="${HF_HOME:-${OPENPI_DATA_HOME}/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"

# Runtime behavior.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Input and memory optimizations.
export XVLA_DYNAMIC_PADDING="${XVLA_DYNAMIC_PADDING:-1}"
export XVLA_DROP_IMAGE_KEYS="${XVLA_DROP_IMAGE_KEYS:-right_wrist_0_rgb}"
export OPENPI_GRADIENT_CHECKPOINTING="${OPENPI_GRADIENT_CHECKPOINTING:-gemma}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29520}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EXP_NAME="${EXP_NAME:-pi05_libero_xpu_8card_input_opt}"
ASSETS_DIR="${ASSETS_DIR:-gs://openpi-assets/checkpoints/pi05_libero/assets}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"
OVERWRITE="${OVERWRITE:-0}"

args=(
  pi05_libero
  --exp-name "${EXP_NAME}"
  --batch-size "${BATCH_SIZE}"
  --data.assets.assets-dir "${ASSETS_DIR}"
)

if [[ "${WANDB_ENABLED}" == "1" ]]; then
  args+=(--wandb-enabled)
else
  args+=(--no-wandb-enabled)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi

# Additional train_pytorch.py options can be passed directly to this script.
args+=("$@")

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log}"

echo "Training log: ${LOG_FILE}"
echo "Config: gpus=${NUM_GPUS} global_batch=${BATCH_SIZE} checkpointing=${OPENPI_GRADIENT_CHECKPOINTING} dynamic_padding=${XVLA_DYNAMIC_PADDING} drop_images=${XVLA_DROP_IMAGE_KEYS}"

torchrun \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  scripts/train_pytorch.py \
  "${args[@]}" \
  2>&1 | tee "${LOG_FILE}"
