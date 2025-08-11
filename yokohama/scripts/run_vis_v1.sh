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


FULL_DETECTION_DIR="yokohama/${place}/full_detection"
DETECTION_PLOT_DIR="yokohama/${place}/detection_plot"
SAVE_DIR="yokohama/${place}/vis"

echo $place

# echo "create graph"
# python yokohama/visualize/create_graph.py --save_dir $SAVE_DIR --source_dir $FULL_DETECTION_DIR --place $place

echo "create movie"
python yokohama/visualize/create_movie.py --save_dir $SAVE_DIR --source_dir $DETECTION_PLOT_DIR --place $place --v_devide_num 2 --h_devide_num 4
# python yokohama/visualize/create_movie.py --save_dir $SAVE_DIR --source_dir $DETECTION_PLOT_DIR --place $place --full

# echo "create movie with graph"
# python yokohama/visualize/create_movie_with_graph.py --save_dir $SAVE_DIR --detection_img_dir $DETECTION_PLOT_DIR --detection_data_dir $FULL_DETECTION_DIR --place $place
