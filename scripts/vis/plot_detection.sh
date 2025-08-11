#!/bin/bash
#SBATCH -p cpu1
#SBATCH -o /homes/hnakayama/congestion_analysis/log/%x-%j.out
#SBATCH -e /homes/hnakayama/congestion_analysis/log/%x-%j.out

source ~/miniconda3/bin/activate hnakayama2

WORKDIR="/homes/hnakayama/congestion_analysis"
cd $WORKDIR

# IMG_DIR=demo/img
# DET_DIR=demo/full_detection
# SAVE_DIR=demo/detection_plot

# IMG_DIR=error_analysis/img
# DET_DIR=error_analysis/full_detection
# SAVE_DIR=error_analysis/detection_plot

IMG_DIR=$1
DET_DIR=$2
SAVE_DIR=$3

echo "Displaying detection data..."
python visualize/detection/plot_detection.py \
        --detection_dir ${DET_DIR} \
        --img_dir ${IMG_DIR} \
        --save_dir ${SAVE_DIR}

echo "Done"
echo "Time: $SECONDS seconds"