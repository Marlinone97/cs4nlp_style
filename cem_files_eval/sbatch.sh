#!/bin/bash

#SBATCH --time=60:00
#SBATCH --account=csnlp_jobs
#SBATCH --output=training.out
#SBATCH --gpus=8
. /etc/profile.d/modules.sh
module add cuda/12.6

python3 robertatraining.py 
