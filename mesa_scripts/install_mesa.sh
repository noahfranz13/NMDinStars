#!/bin/bash

# set necessary environment variables
export MESASDK_ROOT=~/mesasdk
source $MESASDK_ROOT/bin/mesasdk_init.sh

export MESA_DIR=/home/nfranz/mesa-r12778
export OMP_NUM_THREADS=8

# run the install command
cd $MESA_DIR
./install
