#!/bin/bash
#SBATCH -p gpu2
#SBATCH -o ~/congestion_analysis/log/%x-%j.out
#SBATCH -e ~/congestion_analysis/log/%x-%j.out

source ~/miniconda3/bin/activate hnakayama2

WORKDIR="~/congestion_analysis"
cd $WORKDIR

# Input
PATH2VIDEO=$1
SAVE_DIR=$2
WEIGHT_PATH=$3
BEV_FILE=$4
SIZE_FILE=$5

# Frame
IMG_DIR="$SAVE_DIR/img"
# Patch Detection
PATCH_OUT_DIR="$SAVE_DIR/patch_detection"
# Full Detection
FULL_OUT_DIR="$SAVE_DIR/full_detection"
# Tracking
TRACK_DIR="$SAVE_DIR/track"
# Crowd Risk Score
RISK_OUTPUT_DIR="$SAVE_DIR/crowd_risk_score"

# Time Log
TIME_LOG_FILE="$SAVE_DIR/time_log.txt"

echo "Get Frame"
start_time=$(date +%s)
python get_all_frame.py --path2video $PATH2VIDEO --save_dir $IMG_DIR
end_time=$(date +%s)
echo "Get frame time: $((end_time - start_time)) seconds" >> $TIME_LOG_FILE

echo "Detection"
start_time=$(date +%s)
bash scripts/p2pnet/run_inference.sh $PATCH_OUT_DIR $WEIGHT_PATH $IMG_DIR $FULL_OUT_DIR
end_time=$(date +%s)
echo "Detection time: $((end_time - start_time)) seconds" >> $TIME_LOG_FILE
# bash scripts/vis/plot_detection.sh

echo "Tracking"
start_time=$(date +%s)
bash scripts/bytetrack/run_byte_track.sh $FULL_OUT_DIR $TRACK_DIR
end_time=$(date +%s)
echo "Tracking time: $((end_time - start_time)) seconds" >> $TIME_LOG_FILE
# bash scripts/vis/display_track.sh

echo "Crowd Risk Score"
start_time=$(date +%s)
bash scripts/crowd_risk_score/run_crs.sh $TRACK_DIR $RISK_OUTPUT_DIR $BEV_FILE $SIZE_FILE
end_time=$(date +%s)
echo "Crowd Risk Score time: $((end_time - start_time)) seconds" >> $TIME_LOG_FILE
# bash scripts/vis/display_heatmap.sh

echo "Done: $(date +%Y%m%d%H%M%S)"
echo "Total time: $SECONDS seconds"