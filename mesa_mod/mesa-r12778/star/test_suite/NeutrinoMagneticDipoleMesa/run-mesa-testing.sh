#!/bin/bash

# run all 4 processes with M=1.2
# z=0.03, mu=0
#sbatch mesa-testing.slurm 0.03 0 1.2 --job-name=test1

# z=0.02, mu=0
#sbatch mesa-testing.slurm 0.02 0 1.2 --job-name=test2

# z=0.03, mu=0
sbatch mesa-testing.slurm 0.03 6 1.2 --job-name=test3

# z=0.02, mu=0
#sbatch mesa-testing.slurm 0.02 6 1.2 --job-name=test3


# run all 4 processes with M=2
# z=0.03, mu=0
#sbatch mesa-testing.slurm 0.03 0 2 --job-name=test1

# z=0.02, mu=0
#sbatch mesa-testing.slurm 0.02 0 2 --job-name=test2

# z=0.03, mu=0
sbatch mesa-testing.slurm 0.03 6 2 --job-name=test3

# z=0.02, mu=0
#sbatch mesa-testing.slurm 0.02 6 2 --job-name=test3
