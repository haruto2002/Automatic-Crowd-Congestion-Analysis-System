#!/bin/bash
#SBATCH -p gpu2
#SBATCH -o /homes/hnakayama/congestion_analysis/gomi/%x-%j.out
#SBATCH -e /homes/hnakayama/congestion_analysis/gomi/%x-%j.out

source ~/miniconda3/bin/activate P2P

WORKDIR="/homes/hnakayama/congestion_analysis/p2pnet"
cd $WORKDIR

Dataset=$1


python src/main.py p2p default.epochs=500 \
                        dataset.name=$Dataset \
                        optimizer.batch_size.train=8 \
                        optimizer.batch_size.test=2 \