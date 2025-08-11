#!/bin/bash
#SBATCH -p gpu2
#SBATCH -o /homes/hnakayama/congestion_analysis/log/%x-%j.out
#SBATCH -e /homes/hnakayama/congestion_analysis/log/%x-%j.out

source ~/miniconda3/bin/activate P2P

WORKDIR="/homes/hnakayama/congestion_analysis"
cd $WORKDIR

Dataset=$1


python p2pnet/main.py p2p default.epochs=500 \
                        dataset.name=$Dataset \
                        optimizer.batch_size.train=8 \
                        optimizer.batch_size.test=2 \