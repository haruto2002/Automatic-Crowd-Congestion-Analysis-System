#!/bin/bash
#SBATCH -p gpu1
#SBATCH -o /homes/hnakayama/congestion_analysis/log/%x-%j.out
#SBATCH -e /homes/hnakayama/congestion_analysis/log/%x-%j.out

source ~/miniconda3/bin/activate hnakayama2

WORKDIR="/homes/hnakayama/congestion_analysis"
cd $WORKDIR

# place="WorldPorter"
# place="Akarenga"
# place="Chosha"
place="Kokusaibashi"

IMG_DIR="yokohama/${place}/img"
DETECTION_DATA_DIR="yokohama/${place}/full_detection"
SAVE_DIR="yokohama/${place}/vis"

echo $place

echo "create movie with detection and graph"
python yokohama/visualize/create_det_and_graph.py --save_dir $SAVE_DIR --img_dir $IMG_DIR --detection_data_dir $DETECTION_DATA_DIR --resize_ratio 0.25
