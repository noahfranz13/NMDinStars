#!/bin/bash
##SBATCH --job-name=test

#SBATCH --partition=shared
## 3 day max run time for public partitions, except 4 hour limit  in sandbox
#SBATCH --time=0-04:30:00 ## time format is DD-HH:MM:SS

#SBATCH --nodes=1
##SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G ## max amount of memory per node you require
##SBATCH --core-spec=0 ## Uncomment to allow jobs to request all cores on a node
#SBATCH --exclude=lmem-00[01-10]   

#SBATCH --error=error-%A_%a.err ## %A - filled with jobid
#SBATCH --output=output-%A_%a.out ## %A - filled with jobid

## Useful for remote notification
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=user@test.org

## All options and environment variables found on schedMD site: http://slurm.schedmd.com/sbatch.html

mass=1.2
z=$1
y=0.25
mu=$2

uqCode="M${mass}_Y${y}_Z${z}_U${mu}"

## Make new inlist based on template
inlist=inlist_temp_${uqCode}
cp inlist_template $inlist

## write variables to new inlist
sed -i -e "48s/initial_mass =/initial_mass = ${mass}d0/g" $inlist
sed -i -e "49s/initial_z =/initial_z = ${z}d0/g" $inlist
sed -i -e "50s/initial_y =/initial_y = ${y}d0/g" $inlist
sed -i -e "51s/x_ctrl(1) =/x_ctrl(1) = ${mu}d0/g" $inlist
sed -i -e "52s/Zbase =/zbase = ${z}d0/g" $inlist

## Give inlist a unique output name
sed -i -e "88s/nmdm/nmdm_$uqCode/g" $inlist

## make output directory
outpath=/home/nfranz/lus_scratch/mesa-testing
mkdir $outpath

## Run MESA
export MESASDK_ROOT=/home/nfranz/NMDinStars/mesa_mod/mesasdk
source $MESASDK_ROOT/bin/mesasdk_init.sh

export MESA_DIR=/home/nfranz/NMDinStars/mesa_mod/mesa-r12778

export OMP_NUM_THREADS=$([[ -v SLURM_CPUS_PER_TASK ]] && echo ${SLURM_CPUS_PER_TASK} || echo "1")

./star $inlist > terminal_output_${uqCode}.txt

## Move some files around
rm $inlist
mv LOGS/nmdm_${uqCode}.data $outpath
mv terminal_output_${uqCode}.txt $outpath

