#!/bin/bash
#SBATCH -p cpu1
#SBATCH -o /homes/hnakayama/congestion_analysis/log/%x-%j.out
#SBATCH -e /homes/hnakayama/congestion_analysis/log/%x-%j.out

source ~/miniconda3/bin/activate hnakayama2

WORKDIR="/homes/hnakayama/congestion_analysis"
cd $WORKDIR

# place="WorldPorter"
place="Akarenga"
# place="Chosha"
# place="Kokusaibashi"

VIDEO_DIR="/homes/SHARE/Hanabi/20250602_Yokohama/8K/${place}"
SAVE_DIR="yokohama/${place}/img"


python yokohama/get_frames.py \
    --save_dir $SAVE_DIR \
    --video_dir $VIDEO_DIR \
    --freq 60