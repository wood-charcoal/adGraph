#!/bin/bash
#SBATCH -J adGraph
#SBATCH -p kshdexcluxd
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o %j.loop
#SBATCH -e %j.loop

echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"

./build.sh

date
