#!/bin/bash
#SBATCH -p gpu2
#SBATCH -o /homes/hnakayama/congestion_analysis/log/%x-%j.out
#SBATCH -e /homes/hnakayama/congestion_analysis/log/%x-%j.out

source ~/miniconda3/bin/activate hnakayama2

WORKDIR="/homes/hnakayama/congestion_analysis"
cd $WORKDIR

# place="WorldPorter"
place="Akarenga"
# place="Chosha"
# place="Kokusaibashi"

PATCH_OUT_DIR="yokohama/${place}/patch_detection"
WEIGHT_PATH="cutout.pth"
FULL_OUT_DIR="yokohama/${place}/full_detection"
IMG_DIR="yokohama/${place}/img"
SAVE_DIR="yokohama/${place}/detection_plot"

TIME_LOG_FILE="yokohama/${place}/time_log.txt"

start_time=$(date +%s)
bash scripts/p2pnet/run_inference.sh $PATCH_OUT_DIR $WEIGHT_PATH $IMG_DIR $FULL_OUT_DIR
end_time=$(date +%s)
echo "Detection time: $((end_time - start_time)) seconds" >> $TIME_LOG_FILE

start_time=$(date +%s)
bash scripts/vis/plot_detection.sh $IMG_DIR $FULL_OUT_DIR $SAVE_DIR

echo "Done: $(date +%Y%m%d%H%M%S)"
echo "Total time: $SECONDS seconds"