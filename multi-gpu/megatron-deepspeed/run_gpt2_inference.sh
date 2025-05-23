#!/bin/bash
set -ex

BASE_PATH=~/NIR/multi-gpu/megatron-deepspeed/gpt2
DS_CONFIG=ds_config.json

# Type Parallelism
TP=1
PP=1
EP=1

# Model Architecture
NLAYERS=10
HIDDEN=4096
NHEADS=32
SEQ_LENGTH=1024
FFN_HIDDEN_SIZE=4096
MAX_POSITION_EMBEDDINGS=2048
NEXPERTS=1

# Batch
GLOBAL_BATCH=8
MICRO_BATCH=4

GEN_LEN=3
LOOP=3

# DeepSpeed
ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --no-pipeline-parallel ${ds_args}" 
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"

#Logger
LOGGER=logger.log


deepspeed --num_gpus 2 inference_gpt2.py \
    --data-dir "~/NIR/benchmark_full.txt" \
    --loops $LOOP \
    --gen-len $GEN_LEN \
    --output-file "results/DP/dp$NLAYERS.log" \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
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
    --fp16 $ds_args &> $LOGGER


# deepspeed --num_gpus 2 inference_gpt2.py \
#     --data-dir "~/NIR/benchmark_full.txt" \
#     --loops $LOOP \
#     --gen-len $GEN_LEN \
#     --output-file "results/moe/moe{$NLAYERS}.log" \
#     --tensor-model-parallel-size $TP \
#     --pipeline-model-parallel-size $PP \
#     --moe-expert-parallel-size $EP \
#     --num-experts $NEXPERTS \
#     --create-moe-param-group \
#     --num-layers $NLAYERS \
#     --hidden-size $HIDDEN \
#     --num-attention-heads $NHEADS \
#     --seq-length $SEQ_LENGTH \
#     --ffn-hidden-size $FFN_HIDDEN_SIZE \
#     --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
#     --micro-batch-size $MICRO_BATCH \
#     --global-batch-size $GLOBAL_BATCH \
#     --log-interval 1 \
#     --vocab-file $BASE_PATH/vocab.json \
#     --merge-file $BASE_PATH/merges.txt \
#     --inference \
#     --fp16 $ds_args &> $LOGGER


