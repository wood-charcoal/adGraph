#!/bin/bash
#SBATCH -J HIPEnvTest
#SBATCH -p kshdexcluxd
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -o %j-%x.log
#SBATCH -e %j-%x.log
#SBATCH --gres=dcu:1

echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"

cd ${SLURM_SUBMIT_DIR}

module purge
module load compiler/gcc/8.2.0
module load compiler/dtk/24.04
module load compiler/cmake/3.23.3
module list

ulimit -c unlimited

export DTK_ROOT=/public/software/compiler/rocm/dtk-24.04
export PATH="$DTK_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$DTK_ROOT/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=/public/software/compiler/gcc-8.2.0/isl-0.18/lib:$LD_LIBRARY_PATH
export HIP_PATH="$DTK_ROOT"

export CC="$DTK_ROOT/llvm/bin/clang"
export CXX="$DTK_ROOT/bin/hipcc"
export HIPCC="$DTK_ROOT/bin/hipcc"

TEST_FILE=hip_env_test.cpp
OUTPUT_NAME=hip_env_test.out

echo "Compiling hip env test..."

hipcc -std=c++14 ${TEST_FILE} -g -o ${OUTPUT_NAME} \
    -I${DTK_ROOT}/include \
    -I${DTK_ROOT}/include/hip \
    -lhipblas -lrocsparse -lrocrand -lrocsolver -lhipsparse -lamdhip64 -lpthread
    
if [ $? -eq 0 ]; then
    echo "${OUTPUT_NAME} compiled successfully."
else
    echo "Compilation failed for ${TEST_FILE}."
    exit 1
fi

echo "Running hip env test..."

./${OUTPUT_NAME}
if [ $? -eq 0 ]; then
    echo "--- Test ${OUTPUT_NAME} passed. ---"
else
    echo "--- Test ${OUTPUT_NAME} failed. ---"
fi