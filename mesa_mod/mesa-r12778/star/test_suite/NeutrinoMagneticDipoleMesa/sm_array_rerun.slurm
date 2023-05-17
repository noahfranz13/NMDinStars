#!/bin/bash
#SBATCH --time=0-5:00:00 ## time format is DD-HH:MM:SS

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G ## max amount of memory per node you require
#SBATCH --exclude=lmem-00[01-10]   

idx=2228
mass_idx=8
y_idx=9
z_idx=3
mass=1.585714285714286
y=0.300000000000000
z=0.000028200544831

uqCode="M${mass_idx}_Y${y_idx}_Z${z_idx}_i${idx}"

## Make new inlist based on template
inlist=inlist_temp_${uqCode}
cp inlist_template_SM $inlist

## write variables to new inlist
sed -i -e "48s/initial_mass =/initial_mass = ${mass}d0/g" $inlist
sed -i -e "49s/initial_z =/initial_z = ${z}d0/g" $inlist
sed -i -e "50s/initial_y =/initial_y = ${y}d0/g" $inlist
sed -i -e "52s/Zbase =/zbase = ${z}d0/g" $inlist

## Give inlist a unique output name
sed -i -e "88s/nmdm/nmdm_$uqCode/g" $inlist

## make output directory
outpath=/home/nfranz/lus_scratch/sm/out_${uqCode}/
mkdir $outpath

## Run MESA
export MESASDK_ROOT=/home/nfranz/NMDinStars/mesa_mod/mesasdk
source $MESASDK_ROOT/bin/mesasdk_init.sh

export MESA_DIR=/home/nfranz/NMDinStars/mesa_mod/mesa-r12778

export OMP_NUM_THREADS=$([[ -v SLURM_CPUS_PER_TASK ]] && echo ${SLURM_CPUS_PER_TASK} || echo "1")

./star $inlist > terminal_output_${uqCode}.txt

## Move some files around
rm $inlist
mv LOGS/sm_${uqCode}.data $outpath
mv terminal_output_${uqCode}.txt $outpath
