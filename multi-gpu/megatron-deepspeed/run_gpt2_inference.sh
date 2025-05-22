#!/bin/bash
set -ex

BASE_PATH=/home/victor/NIR/multi-gpu/megatron-deepspeed/gpt2
DS_CONFIG=ds_config.json

# Type Parallelism
TP=2
PP=1

# Model Architecture
NLAYERS=24
HIDDEN=1024
NHEADS=16
SEQ_LENGTH=512
FFN_HIDDEN_SIZE=1024
MAX_POSITION_EMBEDDINGS=2048
NEXPERTS=16

# Batch
GLOBAL_BATCH=16
MICRO_BATCH=8

# DeepSpeed
ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --no-pipeline-parallel ${ds_args}" 
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
# ds_args=" --zero-stage=3 ${ds_args}"

#Logger
LOGGER=logger.log


deepspeed --num_gpus 2 inference_gpt2.py \
    --data-dir "/home/victor/NIR/benchmark_full.txt" \
    --loops 3 \
    --gen-len 3 \
    --output-file results.log \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --moe-expert-parallel-size 2 \
    --num-experts $NEXPERTS \
    --create-moe-param-group \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LENGTH \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --log-interval 1 \
    --vocab-file $BASE_PATH/vocab.json \
    --merge-file $BASE_PATH/merges.txt \
    --inference \
    $ds_args &> $LOGGER


