#!/bin/bash
#SBATCH -p cpu1
#SBATCH -o /homes/hnakayama/congestion_analysis/log/%x-%j.out
#SBATCH -e /homes/hnakayama/congestion_analysis/log/%x-%j.out

source ~/miniconda3/bin/activate hnakayama2

WORKDIR="/homes/hnakayama/congestion_analysis"
cd $WORKDIR

TRACK_DIR=demo/track
IMG_DIR=demo/img
VIS_DIR=demo/track_vis

echo "Displaying track data..."
python visualize/tracking/display_track.py \
        --track_dir ${TRACK_DIR} \
        --img_dir ${IMG_DIR} \
        --save_base_dir ${VIS_DIR}

echo "Done"
echo "Time: $SECONDS seconds"