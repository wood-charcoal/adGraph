#!/bin/bash
#SBATCH -J adGraph
#SBATCH -p kshdexcluxd
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o %j-%x.loop
#SBATCH -e %j-%x.loop

echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"

cd ${SLURM_SUBMIT_DIR}

module purge
module load compiler/gcc/8.2.0
module load compiler/rocm/dtk-23.04
module load nvidia/cuda/11.3
module load compiler/cmake/3.23.3
module list

source ./setup_hip_env.sh

# export BUILD_TYPE=Release
# export BUILD_ABI=ON

# export DTK_ROOT=/public/software/compiler/rocm/dtk-23.04
# export CXX=/public/software/compiler/gcc-8.2.0/bin/g++
# export CUDA_TOOLKIT_ROOT_DIR=/public/software/compiler/nvidia/cuda/11.3.1
# export CMAKE_CUDA_COMPILER=/public/software/compiler/nvidia/cuda/11.3.1/bin/nvcc
# export PATH=/public/software/compiler/gcc-8.2.0/bin:$PATH
export LD_LIBRARY_PATH=/public/software/compiler/gcc-8.2.0/isl-0.18/lib:$LD_LIBRARY_PATH

./build_hip.sh -v

date
