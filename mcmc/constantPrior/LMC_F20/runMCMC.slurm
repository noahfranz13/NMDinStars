#!/bin/bash
#SBATCH --job-name=LM_F20
#SBATCH --partition=shared
#SBATCH --time=0-20:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=20G
#SBATCH --exclude=lmem-00[01-10]   

#SBATCH --error=mcmc-%A.err
#SBATCH --output=mcmc-%A.out

#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=nfranz@hawaii.edu

module load lang/Anaconda3
source activate useTf
python3 ../../mcmcNMDM.py --obs LMC_F20
python3 ../../analyzeMCMC.py
