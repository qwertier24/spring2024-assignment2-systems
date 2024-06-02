#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=2
#SBATCH --mem=8G
#SBATCH --time=00:03:00
#SBATCH --gpus-per-task=1


eval "$(conda shell.bash hook)"
# Change conda environment name, if necessary
conda activate cs336
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_PORT: ${MASTER_PORT}"
echo "MASTER_ADDR: ${MASTER_ADDR}"

srun nvidia-smi -L

srun python cs336-systems/cs336_systems/naive_distribution_training.py \
--num_steps=100

# --vocab_set=/home/yukunma_google_com/data/TinyStoriesV2-GPT4-vocab.json \
# --merges_set=/home/yukunma_google_com/data/TinyStoriesV2-GPT4-merges.txt \
# --train_set=/home/yukunma_google_com/data/TinyStoriesV2-GPT4-train.txt \