#!/bin/bash
#SBATCH --job-name=postPro-sm
#SBATCH --partition=shared
#SBATCH --time=0-04:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=19
#SBATCH --mem=16G

#SBATCH --error=post-%A.err
#SBATCH --output=post-%A.out

#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=nfranz@hawaii.edu


module load lang/Anaconda3
source activate useTf
cd /home/nfranz/py_mesa_reader/
pip install .
cd -

./postProcess /home/nfranz/lus_scratch/sm/
