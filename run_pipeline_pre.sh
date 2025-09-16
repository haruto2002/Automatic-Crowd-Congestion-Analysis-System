#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=3:00:00
#PBS -P gaa50073
#PBS -e /home/aag16599bn/research/Automatic-Crowd-Congestion-Analysis-System/log/
#PBS -o /home/aag16599bn/research/Automatic-Crowd-Congestion-Analysis-System/log/

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR

PATH2VIDEO=${PATH2VIDEO:-$1}
SAVE_DIR=${SAVE_DIR:-$2}
WEIGHT_PATH=${WEIGHT_PATH:-$3}
BEV_FILE=${BEV_FILE:-$4}
SIZE_FILE=${SIZE_FILE:-$5}
DET_FREQ=${DET_FREQ:-$6}
TRACK_FREQ=${TRACK_FREQ:-$7}

# Frame
IMG_DIR="${SAVE_DIR}/img"
# Detection
DET_OUT_DIR="${SAVE_DIR}/detection"
FULL_DET_DIR="${DET_OUT_DIR}/frame_detection"
# Tracking
TRACK_DIR="${SAVE_DIR}/track"
# Crowd Risk Score
RISK_OUTPUT_DIR="${SAVE_DIR}/crowd_risk_score"

# Detection Vis
DETECTION_VIS_DIR="${SAVE_DIR}/detection_vis_${DET_FREQ}"
# Tracking Vis
TRACK_VIS_DIR="${SAVE_DIR}/track_vis_${TRACK_FREQ}"

# Time Log
TIME_LOG_FILE="${SAVE_DIR}/time_log.txt"

echo "Get Frame"
start_time=$(date +%s)
bash scripts/get_frame/run_get_frame.sh $PATH2VIDEO $IMG_DIR
end_time=$(date +%s)
echo "Get frame time: $((end_time - start_time)) seconds" >> $TIME_LOG_FILE

echo "Detection"
start_time=$(date +%s)
bash scripts/p2pnet/run_fullsize_inference.sh $WEIGHT_PATH $IMG_DIR $DET_OUT_DIR $FULL_DET_DIR
end_time=$(date +%s)
echo "Detection time: $((end_time - start_time)) seconds" >> $TIME_LOG_FILE


echo "Tracking"
start_time=$(date +%s)
bash scripts/bytetrack/run_byte_track.sh $FULL_DET_DIR $TRACK_DIR
end_time=$(date +%s)
echo "Tracking time: $((end_time - start_time)) seconds" >> $TIME_LOG_FILE

# echo "Crowd Risk Score"
# start_time=$(date +%s)
# bash scripts/crowd_risk_score/run_crs.sh $TRACK_DIR $RISK_OUTPUT_DIR $BEV_FILE $SIZE_FILE
# end_time=$(date +%s)
# echo "Crowd Risk Score time: $((end_time - start_time)) seconds" >> $TIME_LOG_FILE

bash scripts/vis/plot_detection.sh $IMG_DIR $FULL_DET_DIR $DETECTION_VIS_DIR $DET_FREQ
bash scripts/vis/display_track.sh $TRACK_DIR $IMG_DIR $TRACK_VIS_DIR $TRACK_FREQ
# bash scripts/vis/display_heatmap.sh
echo "Visualization time: $((end_time - start_time)) seconds" >> $TIME_LOG_FILE


echo "Done: $(date +%Y%m%d%H%M%S)"
echo "Total time: $SECONDS seconds"