#!/bin/bash
#SBATCH -p gpu1
#SBATCH -o /homes/hnakayama/P2P/P2PNet/gomi/%x-%j.out
#SBATCH -e /homes/hnakayama/P2P/P2PNet/gomi/%x-%j.out

source ~/miniconda3/bin/activate hnakayama2
cd /homes/hnakayama/P2P/P2PNet

save_dir=vis/plot
weight_path=$1
weight_name=$2
dataset_name=$3
resource=local
# resource=Tsukuba
gpu_id=0

echo $dataset_name
echo predicting
python src/run_predict.py $save_dir $weight_path $weight_name $dataset_name $resource $gpu_id

save_dir=vis
source_dir=vis

echo making graph
python vis_codes/create_graph.py $save_dir $source_dir $dataset_name $weight_name

echo making movie
python vis_codes/create_mov_with_time.py $save_dir $source_dir $dataset_name $weight_name

echo making movie_graph
python vis_codes/create_movie_graph.py $save_dir $source_dir $dataset_name $weight_name