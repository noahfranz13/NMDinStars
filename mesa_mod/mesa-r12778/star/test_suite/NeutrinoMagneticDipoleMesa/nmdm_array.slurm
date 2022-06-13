#!/bin/bash
##SBATCH --job-name=timingArray

##SBATCH --array=1-1296

#SBATCH --partition=kill-shared
## 3 day max run time for public partitions, except 4 hour limit  in sandbox
#SBATCH --time=0-05:00:00 ## time format is DD-HH:MM:SS

#SBATCH --nodes=1
##SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G ## max amount of memory per node you require
##SBATCH --core-spec=0 ## Uncomment to allow jobs to request all cores on a node
#SBATCH --exclude=lmem-00[01-10]   

##SBATCH --error=hello-%A_%a.err ## %A - filled with jobid
##SBATCH --output=hello-%A_%a.out ## %A - filled with jobid

## Useful for remote notification
##SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
##SBATCH --mail-user=user@test.org

## All options and environment variables found on schedMD site: http://slurm.schedmd.com/sbatch.html
## Read in line of grid and save each as a variable
line=($(sed -n "$SLURM_ARRAY_TASK_ID p" $1))
mass_idx=${line[0]}
y_idx=${line[1]}
z_idx=${line[2]}
mu_idx=${line[3]}
mass=${line[4]}
y=${line[5]}
z=${line[6]}
mu=${line[7]}

uqCode=${mass_idx}_${y_idx}_${z_idx}_${mu_idx}

## Make new inlist based on template
inlist=inlist_temp_${uqCode}
cp inlist_template $inlist

## write variables to new inlist
sed -i -e "48s/initial_mass =/initial_mass = $mass/g" $inlist
sed -i -e "49s/initial_z =/initial_z = $z/g" $inlist
sed -i -e "50s/initial_y =/initial_y = $y/g" $inlist
sed -i -e "51s/x_ctrl(1) =/x_ctrl(1) = $mu/g" $inlist
sed -i -e "52s/Zbase =/zbase = $z/g" $inlist

## Give inlist a unique output name
sed -i -e "88s/nmdm/nmdm_$uqCode/g" $inlist

## Run MESA
export MESASDK_ROOT=/home/nfranz/NMDinStars/mesa_mod/mesasdk
source $MESASDK_ROOT/bin/mesasdk_init.sh

export MESA_DIR=/home/nfranz/NMDinStars/mesa_mod/mesa-r12778

export OMP_NUM_THREADS=$([[ -v SLURM_CPUS_PER_TASK ]] && echo ${SLURM_CPUS_PER_TASK} || echo "1")

./star $inlist > terminal_output_${uqCode}.txt

## Move some files around
rm $inlist
outpath=/home/nfranz/lus_scratch/mesa-1296/out_${uqCode}/
mkdir $outpath
mv LOGS/nmdm_${uqCode}.data $outpath
mv terminal_output_${uqCode}.txt $outpath
