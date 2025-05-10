#!/bin/sh

LOG_FILE="logger_config.log"

BATCH_SIZE=2
HIDDEN_SIZE=1024

USE_ZERO=1
USE_CPU_OFFLOAD=1
PIN_MEMORY=1
USE_DISK_OFFLOAD=1
USE_AIO=1
BUFFER_COUNT=2
BUFFER_SIZE=0.5
OFFLOAD_DIR="offload"
USE_GDS=0 # not supported on my platform

USE_QUANT=0
QUANT_BITS=0
QUANT_GROUP_SIZE=0

HALF_PRECISION="not" 

ARGUMENTS="--batch-size $BATCH_SIZE --hidden-size $HIDDEN_SIZE"

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

if [ $USE_QUANT -eq 1 ]; then
    ARGUMENTS="$ARGUMENTS --quant-bits $QUANT_BITS --quant-group-size $QUANT_GROUP_SIZE"
fi

echo $ARGUMENTS
python3 Configurator.py $ARGUMENTS &> $LOG_FILE