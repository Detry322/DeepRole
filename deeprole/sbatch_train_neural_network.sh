#!/bin/bash
#SBATCH --job-name=train_deeprole_nn
#SBATCH --output=sbatch_logs/train_deeprole_nn-%j.out
#SBATCH --error=sbatch_logs/train_deeprole_nn-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --time=0:30:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jserrino

NS="$1"
NF="$2"
PC="$3"

module add openmind/singularity
singularity exec --nv -B /om:/om /om/user/jserrino/singularity/1.13.0rc1-gpu.img ./train_nn.sh $NS $NF $PC
