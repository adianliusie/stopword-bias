#!/bin/bash
#$ -S /bin/bash

source ~/.bashrc
conda activate torch1.8

unset LD_PRELOAD
export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE

python ../run_train.py $@

# qsub -cwd -j yes -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='volta' -l osrel='*' -o 'LOGs/run_1.log' run.sh

