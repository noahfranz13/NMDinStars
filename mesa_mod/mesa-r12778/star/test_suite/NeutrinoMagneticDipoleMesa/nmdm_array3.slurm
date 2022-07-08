#!/bin/bash
##SBATCH --job-name=grid1

##SBATCH --array=1-1296

##SBATCH --partition=kill-shared
## 3 day max run time for public partitions, except 4 hour limit  in sandbox
#SBATCH --time=0-04:30:00 ## time format is DD-HH:MM:SS

#SBATCH --nodes=1
##SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G ## max amount of memory per node you require
##SBATCH --core-spec=0 ## Uncomment to allow jobs to request all cores on a node
#SBATCH --exclude=lmem-00[01-10]   

##SBATCH --error=error-%A_%a.err ## %A - filled with jobid
##SBATCH --output=output-%A_%a.out ## %A - filled with jobid

## Useful for remote notification
##SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
##SBATCH --mail-user=user@test.org

## All options and environment variables found on schedMD site: http://slurm.schedmd.com/sbatch.html
## Read in line of grid and save each as a variable
line=($(sed -n "$SLURM_ARRAY_TASK_ID p" $1))
idx=${line[0]}
mass_idx=${line[1]}
y_idx=${line[2]}
z_idx=${line[3]}
mu_idx=${line[4]}
mass=${line[5]}
y=${line[6]}
z=${line[7]}
mu=${line[8]}

uqCode="M${mass_idx}_Y${y_idx}_Z${z_idx}_U${mu_idx}_i${idx}"

## Make new inlist based on template
inlist=inlist_temp_${uqCode}
cp inlist_template $inlist

## write variables to new inlist
sed -i -e "48s/initial_mass =/initial_mass = ${mass}d0/g" $inlist
sed -i -e "49s/initial_z =/initial_z = ${z}d0/g" $inlist
sed -i -e "50s/initial_y =/initial_y = ${y}d0/g" $inlist
sed -i -e "51s/x_ctrl(1) =/x_ctrl(1) = ${mu}d0/g" $inlist
sed -i -e "52s/Zbase =/Zbase = ${z}d0/g" $inlist

## Give inlist a unique output name
sed -i -e "88s/nmdm/nmdm_$uqCode/g" $inlist

## make output directory
outpath=/home/nfranz/lus_scratch/mesa-112500/third/out_${uqCode}/
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
