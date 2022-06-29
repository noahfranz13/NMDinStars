#!/bin/bash
#SBATCH --job-name=postProcess1
#SBATCH --partition=sandbox
#SBATCH --time=0-04:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=19
#SBATCH --mem=16G

#SBATCH --error=post-%A.err
#SBATCH --output=post-%A.out

#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=nfranz@hawaii.edu

./postProcess /home/nfranz/lus_scratch/mesa-112500/first/
