#!/bin/bash
#SBATCH --job-name=nmdm_normal
#SBATCH --partition=shared
## 3 day max run time for public partitions, except 4 hour max runtime for the sandbox partition
#SBATCH --time=0-04:00:00 ## time format is DD-HH:MM:SS

#SBATCH --nodes=1
##SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G ## max amount of memory per node you require
##SBATCH --core-spec=0 ## Uncomment to allow jobs to request all cores on a node
#SBATCH --exclude=lmem-00[01-10]   

#SBATCH --error=error-%A.err ## %A - filled with jobid
#SBATCH --output=MESA-%A.out ## %A - filled with jobid

## Useful for remote notification
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=nfranz@hawaii.edu

#module load compiler/GCCcore/9.2.0
#module load lib/zlib/1.2.11-GCCcore-6.4.0

export MESASDK_ROOT=/home/nfranz/NMDinStars/mesa_mod/mesasdk
source $MESASDK_ROOT/bin/mesasdk_init.sh

export MESA_DIR=/home/nfranz/NMDinStars/mesa_mod/mesa-r12778

export OMP_NUM_THREADS=$([[ -v SLURM_CPUS_PER_TASK ]] && echo ${SLURM_CPUS_PER_TASK} || echo "1")

./star inlist_normal #You could also use ./rn depending on how your inlists are set up.
