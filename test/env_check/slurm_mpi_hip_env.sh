#!/bin/bash
#SBATCH -J adGraphTest
#SBATCH -p kshdexcluxd
#SBATCH -N 2
#SBATCH --ntasks 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:1
#SBATCH -o %j-%x.log
#SBATCH -e %j-%x.log

echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"

cd ${SLURM_SUBMIT_DIR}

module purge
module load compiler/gcc/8.2.0
module load compiler/dtk/24.04
module load compiler/cmake/3.23.3
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4-gcc-7.3.1
module list

ulimit -c unlimited

# ---------------------- set env path ----------------------
DEFAULT_ROCM_PATH="/public/software/compiler/rocm/dtk-24.04"

export DTK_ROOT="$DEFAULT_ROCM_PATH"
export PATH="$DTK_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$DTK_ROOT/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=/public/software/compiler/gcc-8.2.0/isl-0.18/lib:$LD_LIBRARY_PATH
export HIP_PATH="$DTK_ROOT"

export CC="$DTK_ROOT/llvm/bin/clang"
export CXX="$DTK_ROOT/bin/hipcc"
export HIPCC="$DTK_ROOT/bin/hipcc"
export MPICH_GPU_SUPPORT_ENABLED=1
export HSA_ENABLE_SDMA=0 

# ---------------------- compilation ----------------------
echo "--- Starting Compilation ---"

TARGET_ARCH="gfx90a"
MPI_LINK_FLAGS=$(mpicxx -showme:link)

$DTK_ROOT/bin/hipcc mpi_hip_env_test.cpp -o mpi_hip_env_test \
    -std=c++17 \
    --offload-arch=$TARGET_ARCH \
    $MPI_LINK_FLAGS

if [ $? -eq 0 ]; then
    echo "--- Compilation Successful ---"
else
    echo "--- Compilation FAILED ---"
    exit 1
fi

# ---------------------- execution ----------------------
echo "--- Starting Execution (${SLURM_NTASKS} ranks on ${SLURM_NNODES} nodes) ---"

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

mpirun ./mpi_hip_env_test

echo "--- Job Finished ---"
