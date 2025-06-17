#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh"
echo "=============================================================================================================="

EXEC_PATH=$(pwd)

if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

# mpirun -np 2 --output-filename log_output --merge-stderr-to-stdout python tensor.py > tensor.txt
mpirun -np 2 --output-filename log_output --merge-stderr-to-stdout python pipeline.py > pipeline.txt
# mpirun -np 2 --output-filename log_output --merge-stderr-to-stdout python sharding.py > sharding.txt
# mpirun -np 2 --output-filename log_output --merge-stderr-to-stdout python single.py > single.txt