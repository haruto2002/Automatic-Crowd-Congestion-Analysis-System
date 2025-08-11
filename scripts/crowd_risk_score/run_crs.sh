#!/bin/bash
#SBATCH -p cpu1
#SBATCH -o /homes/hnakayama/congestion_analysis/log/%x-%j.out
#SBATCH -e /homes/hnakayama/congestion_analysis/log/%x-%j.out

source ~/miniconda3/bin/activate hnakayama2

WORKDIR="/homes/hnakayama/congestion_analysis"
cd $WORKDIR

# TRACK_DIR="demo/track"
# RISK_OUTPUT_DIR="demo/crowd_risk_score/debug"
# BEV_FILE="crs/bev/WorldPorters_8K_matrix.txt"
# SIZE_FILE="crs/size/WorldPorters_8K_size.txt"

TRACK_DIR=$1
RISK_OUTPUT_DIR=$2
BEV_FILE=$3
SIZE_FILE=$4

python crs/main.py config \
    output_dir=${RISK_OUTPUT_DIR} \
    track_dir=${TRACK_DIR} \
    bev_file=${BEV_FILE} \
    size_file=${SIZE_FILE} \
    frame_range="[null, null]" \


python crs/utils/smoothing.py \
    --source_dir ${RISK_OUTPUT_DIR}/each_result/danger_score \
    --save_dir ${RISK_OUTPUT_DIR}/each_result/danger_score_smoothed \
    --space_window_size 3 \
    --time_window_size 10
