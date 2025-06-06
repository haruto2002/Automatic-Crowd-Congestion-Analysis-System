#!/bin/bash
#$ -l rt_F=2
#$ -l h_rt=24:00:00
#$ -l USE_SSH=1
#$ -v SSH_PORT=2299
#$ -j y
#$ -e /home/aag16599bn/gcc50570/nakayama/congestion_analysis/gomi/
#$ -o /home/aag16599bn/gcc50570/nakayama/congestion_analysis/gomi/
#$ -cwd

umask 007

source /etc/profile.d/modules.sh
source /home/aag16599bn/miniconda3/bin/activate hnakayama

WORKDIR="/home/aag16599bn/gcc50570/nakayama/congestion_analysis/p2pnet"
cd $WORKDIR

# get host list
cat $SGE_JOB_HOSTLIST > $WORKDIR/hostfiles/hostfile_$JOB_ID
HOST=${HOSTNAME:0:5}

# OUT_DIR=$1
# Dataset=$2
# DNN_task=$3
# Project_name=$4
# Run_name=$5

OUT_DIR=outputs_debug/
Dataset=Yokohama_Hanabi_2024-09-01
DNN_task=hanabi
Project_name=debug
Run_name=debug_ABCI

# params
SEED=111
epochs=100
lr=0.1
lr_decay=0.001

PYTHON_COMMAND="python src/main.py p2p out_dir=$OUT_DIR \
                        default.resource=ABCI \
                        default.bar=True \
                        default.r_seed=$SEED \
                        default.epochs=$epochs \
                        default.finetune=True \
                        default.wandb=True \
                        default.multi_node=True \
                        default.work_dir=$WORKDIR \
                        tmp.job_id=$JOB_ID \
                        tmp.master_node=$HOST \
                        tmp.n_node=$NHOSTS \
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
                        wandb.project_name=$Project_name \
                        wandb.run_name=$Run_name"

COMMAND="umask 007; \
        cd $WORKDIR; \
        source /home/aag16599bn/miniconda3/bin/activate hnakayama; \
        $PYTHON_COMMAND"

for hostname in $(cat $SGE_JOB_HOSTLIST);do
    ssh -p 2299 -o "StrictHostKeyChecking=no" $hostname $COMMAND &
    sleep 5
done
wait