#!/bin/bash
#SBATCH --job-name=sch_solub
#SBATCH --partition=cpuPartition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --error=%j.err
#SBATCH --output=%j.out
##########################################
I=`whoami`
CURDIR=`pwd`
rm -rf $CURDIR/nodelist.$SLURM_JOB_ID
NODES=`scontrol show hostnames $SLURM_JOB_NODELIST`
for i in $NODES
do
echo "$i:$SLURM_NTASKS_PER_NODE" >> $CURDIR/nodelist.$SLURM_JOB_ID
done
echo $SLURM_NPROCS

echo "process will start at : "
date
echo "++++++++++++++++++++++++++++++++++++++++"

export PATH="/lustre/home/qfchen/anaconda3/bin:$PATH"
export PATH="/lustre/home/qfchen/anaconda3/envs/RDKit/:$PATH"
export PATH="/lustre/home/qfchen/anaconda3/envs/RDKit/bin/:$PATH"

# source /lustre/home/qfchen/.bashrc
#conda activate RDKit
python  /lustre/home/qfchen/GCN-Prediction/sch_train_solub.py > /lustre/home/qfchen/GCN-Prediction/sch_solub.log

echo "++++++++++++++++++++++++++++++++++++++++"
echo "process end at : "
date
sleep 300
