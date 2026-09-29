#!/bin/bash
set -e
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.."

export NCCL_P2P_LEVEL=NVL
export TOKENIZERS_PARALLELISM=True

SPLIT="${SPLIT:-structural}"
case "$SPLIT" in structural|temporal) ;; *) echo "SPLIT must be structural or temporal" >&2; exit 1 ;; esac

MODEL_PATH="./outputs/DSLlama_${SPLIT}_s2_merged"
CSV_PATH="./data/${SPLIT}_split/test_with_label.csv"
ESM_MODEL_NAME="./esm2_t30_150M_UR50D.pt"
MAX_LENGTH=1024

GPU_ID=0

OUTPUT_DIR="${MODEL_PATH}"
OUTPUT_PATH="${OUTPUT_DIR}/result.jsonl"
mkdir -p "$OUTPUT_DIR"
cp "$SCRIPT_PATH" "${OUTPUT_DIR}/" # Save a copy of the script

CUDA_VISIBLE_DEVICES=${GPU_ID} python infer.py \
    --model_name_or_path "${MODEL_PATH}" \
    --max_length "${MAX_LENGTH}" \
    --csv_path "${CSV_PATH}" \
    --esm_model_name "${ESM_MODEL_NAME}" \
    --output_path "${OUTPUT_PATH}" \
    --do_sample False \
    --ephod_label_col "ephod_label" \
    --pptstab_label_col "pptstab_label"
