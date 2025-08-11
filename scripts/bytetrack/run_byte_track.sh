#!/bin/bash
#SBATCH -p cpu1
#SBATCH -o /homes/hnakayama/congestion_analysis/log/%x-%j.out
#SBATCH -e /homes/hnakayama/congestion_analysis/log/%x-%j.out

source ~/miniconda3/bin/activate hnakayama2

WORKDIR="/homes/hnakayama/congestion_analysis"
cd $WORKDIR

# TRACK_SAVE_DIR="demo/track"
# DETECTION_DIR="demo/full_detection"

DETECTION_DIR=$1
TRACK_SAVE_DIR=$2

python bytetrack/run_point_tracker.py \
        --save_dir ${TRACK_SAVE_DIR} \
        --source_dir ${DETECTION_DIR} \
        --img_h_size 4320 \
        --img_w_size 7680 \
        --track_thresh 0.5 \
        --track_buffer 30 \
        --match_thresh 10.0 \
        --distance_metric euclidean


# sbatch byte_track/scripts/run_byte_track.sh WorldPorters_noon
# sbatch byte_track/scripts/run_byte_track.sh WorldPorters_night
# sbatch byte_track/scripts/run_byte_track.sh WorldPorters_night2
