#!/bin/sh

LOG_FILE="logger.log"
FILE="run_opt_experiment.py"

MODEL_NAME=facebook/opt-1.3b
BATCH_SIZE=4
PROMPT_LEN=512
GEN_LEN=32
LOOPS=3

NUM_GPUS=2

USE_ZERO=0
USE_CPU_OFFLOAD=0
PIN_MEMORY=0
USE_DISK_OFFLOAD=0
USE_AIO=0
BUFFER_COUNT=2
BUFFER_SIZE=0.5
OFFLOAD_DIR="offload"
USE_GDS=0 # not supported on my platform

USE_TP_PARALLEL=1

USE_KV_OFFLOAD=0
PIN_KV_CACHE=0
ASYNC_KV_OFFLOAD=0

USE_QUANT=0
QUANT_BITS=0
QUANT_GROUP_SIZE=0

HALF_PRECISION="not"

LOG_FILE="logger.log"
OUTPUT_FILE="results.log"
VERBOSE=2

ARGUMENTS="--num_gpus $NUM_GPUS $FILE --loops $LOOPS --model $MODEL_NAME --batch-size $BATCH_SIZE --prompt-len $PROMPT_LEN --gen-len $GEN_LEN"

if [ $USE_ZERO -eq 1 ]; then
    ARGUMENTS="$ARGUMENTS --use-zero"
    if [ $USE_CPU_OFFLOAD -eq 1 ]; then
        ARGUMENTS="$ARGUMENTS --cpu-offload"
    fi
    if [ $USE_DISK_OFFLOAD -eq 1 ]; then
        ARGUMENTS="$ARGUMENTS --disk-offload --buffer-count $BUFFER_COUNT --buffer-size $BUFFER_SIZE --offload-dir $OFFLOAD_DIR"
        if [ $USE_AIO -eq 1 ]; then
            ARGUMENTS="$ARGUMENTS --use-aio"
        fi
        if [ $USE_GDS -eq 1 ]; then
            ARGUMENTS="$ARGUMENTS --use-gds"
        fi
    fi
    if [ $PIN_MEMORY -eq 1 ]; then
        ARGUMENTS="$ARGUMENTS --pin-memory"
    fi
fi

if [ $USE_TP_PARALLEL -eq 1 ]; then
    ARGUMENTS="$ARGUMENTS --use-tp-parallel"
fi

ARGUMENTS="$ARGUMENTS --output-file $OUTPUT_FILE --verbose $VERBOSE"

echo "deepspeed $ARGUMENTS"
deepspeed $ARGUMENTS &> $LOG_FILE