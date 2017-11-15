#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --time=72:00:00

module add plgrid/libs/libpng

python ${1}
