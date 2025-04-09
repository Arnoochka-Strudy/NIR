#!/bin/sh

MODEL_NAME=facebook/opt-125m
BATCHSIZE=3
PROMPT_LEN=512
GEN_LEN=32 

USE_CPU_OFFLOAD=1
USE_KV_OFFLOAD=1 
USE_HF_MODEL=0
USE_QUANT=0
USE_DISK_OFFLOAD=0
USE_GDS=1

OFFLOAD_DIR="offload"
LOG_FILE="logger.log"

if [ $USE_CPU_OFFLOAD -eq 1 ]; then
    CPU_OFFLOAD="--cpu-offload"
else
    CPU_OFFLOAD=""
fi

if [ $USE_KV_OFFLOAD -eq 1 ]; then
    KV_OFFLOAD="--kv-offload"
else
    KV_OFFLOAD=""
fi

if [ $USE_HF_MODEL -eq 1 ]; then
    HF_MODEL="--hf-model"
else
    HF_MODEL=""
fi

if [ $USE_QUANT -eq 1 ]; then
    QUANT_BITS="--quant_bits"
else
    QUANT_BITS=""
fi

if [ $USE_DISK_OFFLOAD -eq 1 ]; then
    DISK_OFFLOAD="--disk-offload --offload-dir $OFFLOAD_DIR"
else
   DISK_OFFLOAD=""
fi 

if [ $USE_GDS -eq 1 ]; then
    USE_GDS_FLAG="--use_gds"
else
    USE_GDS_FLAG=""
fi

deepspeed --num_gpus 1 run_model.py \
 --model ${MODEL_NAME} --batch-size ${BATCHSIZE} --prompt-len ${PROMPT_LEN} --gen-len ${GEN_LEN} ${USE_GDS_FLAG} --pin-memory 1 \
 ${CPU_OFFLOAD} ${KV_OFFLOAD} ${DISK_OFFLOAD} ${QUANT_BITS} \
  &> $LOG_FILE