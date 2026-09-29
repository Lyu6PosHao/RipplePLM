#!/bin/bash
set -e
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.."

export NCCL_P2P_LEVEL=NVL
export TOKENIZERS_PARALLELISM=True

SPLIT="${SPLIT:-structural}"
case "$SPLIT" in structural|temporal) ;; *) echo "SPLIT must be structural or temporal" >&2; exit 1 ;; esac

BASE_MODEL_PATH="${BASE_MODEL_PATH:-./DeepSeek-R1-Distill-Llama-8B}"
LORA_R=1
LORA_ALPHA=1
LORA_TARGETS="q_proj"

MODULES_TO_SAVE=mpf
MERGE_WHEN_FINISHED=True

CSV_PATH="./data/${SPLIT}_split/train_with_label.csv"
VAL_CSV_PATH="./data/${SPLIT}_split/val_with_label.csv"
ESM_MODEL_NAME="./esm2_t30_150M_UR50D.pt"
MAX_LENGTH=1024
EPHOD_LABEL_COL='ephod_label'
PPTSTAB_LABEL_COL='pptstab_label'
ENABLE_PROPERTY_TASK=False
EVAL_MAX_SAMPLES=100

EPOCHS=1
TRAIN_BATCH_SIZE=6
GRAD_ACCUMULATION=1
LEARNING_RATE=2e-4

GPUS="0,1,2,3,4,5,6,7"
MASTER_PORT=29501

OUTPUT_DIR="./outputs/DSLlama_${SPLIT}_s1"
mkdir -p "$OUTPUT_DIR"
cp "$SCRIPT_PATH" "${OUTPUT_DIR}/"

deepspeed --include "localhost:${GPUS}" --master_port ${MASTER_PORT} train.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path "${BASE_MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --csv_path "${CSV_PATH}" \
    --val_csv_path "${VAL_CSV_PATH}" \
    --eval_max_samples "${EVAL_MAX_SAMPLES}" \
    --esm_model_name "${ESM_MODEL_NAME}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_targets "${LORA_TARGETS}" \
    --modules_to_save "${MODULES_TO_SAVE}" \
    --merge_when_finished "${MERGE_WHEN_FINISHED}" \
    --max_length "${MAX_LENGTH}" \
    --ephod_label_col "${EPHOD_LABEL_COL}" \
    --pptstab_label_col "${PPTSTAB_LABEL_COL}" \
    --enable_property_task "${ENABLE_PROPERTY_TASK}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUMULATION}" \
    --learning_rate "${LEARNING_RATE}" \
    --bf16 True \
    --do_train \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --logging_steps 8 \
    --warmup_ratio 0.05 \
    --report_to "none"
