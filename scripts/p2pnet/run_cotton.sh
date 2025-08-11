#!/bin/bash
#SBATCH -p gpu2
#SBATCH -o /homes/hnakayama/congestion_analysis/log/%x-%j.out
#SBATCH -e /homes/hnakayama/congestion_analysis/log/%x-%j.out

source ~/miniconda3/bin/activate P2P

WORKDIR="/homes/hnakayama/congestion_analysis"
cd $WORKDIR


# OUT_DIR=$1
# Dataset=$2
# DNN_task=$3
# Project_name=$4
# Run_name=$5

OUT_DIR=outputs_20241219/
Dataset=Yokohama_Hanabi_2023+2024_20241217
DNN_task=hanabi
Project_name=train_for_track
Run_name=20241219

# params
SEED=111
epochs=3000
lr=0.1
lr_decay=0.0001


python p2pnet/main.py p2p out_dir=$OUT_DIR \
                        default.resource=Tsukuba \
                        default.bar=True \
                        default.r_seed=$SEED \
                        default.epochs=$epochs \
                        default.finetune=True \
                        default.wandb=True \
                        default.num_workers=2 \
                        dataset.name=$Dataset \
                        optimizer.batch_size.train=8 \
                        optimizer.batch_size.test=2 \
                        optimizer.scheduler.name=cosine \
                        optimizer.hp.lr=$lr \
                        optimizer.hp.lr_decay=$lr_decay \
                        optimizer.hp.momentum=0.95 \
                        optimizer.hp.weight_decay=1e-4 \
                        optimizer.scheduler.warmup=True \
                        optimizer.hp.lr_warmup_step=10 \
                        optimizer.hp.lr_warmup_init=1e-5 \
                        network.init_weight=SHTechA.pth \
                        network.dnn_task=$DNN_task \
                        network.all_save=True \
                        wandb.project_name=$Project_name \
                        wandb.run_name=$Run_name
