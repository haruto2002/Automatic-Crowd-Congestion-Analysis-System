#!/bin/bash
#SBATCH -p gpu2
#SBATCH -o /homes/hnakayama/congestion_analysis/gomi/%x-%j.out
#SBATCH -e /homes/hnakayama/congestion_analysis/gomi/%x-%j.out

source ~/miniconda3/bin/activate P2P

WORKDIR="/homes/hnakayama/congestion_analysis/p2pnet"
cd $WORKDIR

OUT_DIR=$1
WEIGHT_PATH=$2
IMG_DIR=$3

python src/run_inference_ddp.py p2p \
    out_dir=$OUT_DIR \
    default.finetune=True \
    network.init_weight=$WEIGHT_PATH \
    optimizer.batch_size.test=2 \
    default.bar=True \
    default.num_workers=2 \
    img_dir=$IMG_DIR