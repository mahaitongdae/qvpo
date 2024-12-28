#!/bin/bash
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -t 0-00:40
#SBATCH -p gpu_test
#SBATCH --mem=8G
#SBATCH -o pytorch_%j.out
#SBATCH -e pytorch_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=haitongma@g.harvard.edu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1 
#SBATCH --cpus-per-task=16
#SBATCH --array=1-5%5


# Load software modules and source conda environment
module load python/3.10.12-fasrc01
mamba activate torch39

# qexp without any change
python main.py --env_name HalfCheetah-v3 --weighted --aug --seed 0 --q_transform qexp --use_action_target --cuda cpu

# remove action target
python main.py --env_name HalfCheetah-v4 --weighted --aug --seed 0  --cuda cpu

# remove entropy reg
python main.py --env_name HalfCheetah-v4 --weighted --aug --seed 0 --entropy_alpha 0.0  --cuda cpu
