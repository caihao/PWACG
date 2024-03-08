#!/bin/bash

#SBATCH --account=caihao1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output="{{log_file}}"
#SBATCH --job-name=sean_{{run_num}}

module load singularity

cd $SLURM_SUBMIT_DIR

singularity exec --nv --bind /project/`whoami`,/scratch/`whoami` /scratch/caihao5/singularity/tensorflow.sif python3 {{run_file}}