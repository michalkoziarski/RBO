#!/bin/bash

for i in $( seq 1 $1 )
do
	sbatch slurm.sh run.py
done
