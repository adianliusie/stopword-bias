#!/bin/bash

log="$(date +"%I_%M_%S_%p-%d_%m").log"

qsub -cwd -j yes -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' -o 'LOGs/'$log run.sh $@
# qsub -cwd -j yes -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='pascal' -l osrel='*' -o 'LOGs/'$log run.sh $@
# qsub -cwd -j yes -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='volta' -l osrel='*' -o 'LOGs/'$log run.sh $@
