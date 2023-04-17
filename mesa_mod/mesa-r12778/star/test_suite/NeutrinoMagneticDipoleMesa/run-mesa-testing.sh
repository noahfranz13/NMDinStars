#!/bin/bash

# run all 4 processes
# z=0.03, mu=0
sbatch mesa-testing.slurm 0.03 0 --job-name=test1

# z=0.02, mu=0
sbatch mesa-testing.slurm 0.02 0 --job-name=test2

# z=0.03, mu=0
sbatch mesa-testing.slurm 0.03 6 --job-name=test3

# z=0.02, mu=0
sbatch mesa-testing.slurm 0.02 6 --job-name=test3

# run plotting code
python3 plot-testing.py
