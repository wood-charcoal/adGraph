#!/bin/bash
#SBATCH -J test_single
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

# ---------------------- set env path ----------------------
TEST_CODE_DIR="$HOME/projects/testing/adGraph/test/single_node_test" 
LAPACK_INSTALL_ROOT="$HOME/projects/lapack-3.12.0/install"
NVGRAPH_LIB_DIR="$HOME/projects/testing/adGraph/dist/lib64" 
NVGRAPH_INCLUDE_DIR="$HOME/projects/testing/adGraph/dist/include/nvgraph"

DEFAULT_ROCM_PATH="/public/software/compiler/rocm/dtk-24.04"

export DTK_ROOT="$DEFAULT_ROCM_PATH"
export PATH="$DTK_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$DTK_ROOT/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=/public/software/compiler/gcc-8.2.0/isl-0.18/lib:$LD_LIBRARY_PATH
export HIP_PATH="$DTK_ROOT"

export CC="$DTK_ROOT/llvm/bin/clang"
export CXX="$DTK_ROOT/bin/hipcc"
export HIPCC="$DTK_ROOT/bin/hipcc"

# ---------------------- compile scripts ----------------------

for TEST_FILE in ${TEST_CODE_DIR}/adgraph_test*.cpp; do
    
    BASENAME=$(basename ${TEST_FILE} .cpp)
    OUTPUT_NAME="${BASENAME}.out"

    echo "Compiling ${TEST_FILE} to ${OUTPUT_NAME}..."
    
    hipcc -std=c++14 ${TEST_FILE} -g -o ${OUTPUT_NAME} \
        -I${NVGRAPH_INCLUDE_DIR} \
        -I${LAPACK_INSTALL_ROOT}/include \
        -I${DTK_ROOT}/include \
        -I${DTK_ROOT}/include/hip \
        -L${NVGRAPH_LIB_DIR} \
        -L${LAPACK_INSTALL_ROOT}/lib64 \
        -lnvgraph \
        -lcblas -lblas -llapack \
        -lhipblas -lrocsparse -lrocrand -lrocsolver -lhipsparse -lamdhip64 -lpthread
        
    if [ $? -eq 0 ]; then
        echo "${OUTPUT_NAME} compiled successfully."
    else
        echo "Compilation failed for ${TEST_FILE}."
        exit 1
    fi
done

echo "--------------------------------------------------------"

# ---------------------- run tests ----------------------

export LD_LIBRARY_PATH=${LAPACK_INSTALL_ROOT}/lib64:${NVGRAPH_LIB_DIR}:${LD_LIBRARY_PATH}

echo "Running tests..."

for OUTPUT_FILE in adgraph_test*.out; do
    echo "--- Running ${OUTPUT_FILE} ---"
    ./${OUTPUT_FILE}
    if [ $? -eq 0 ]; then
        echo "--- Test ${OUTPUT_FILE} passed. ---"
    else
        echo "--- Test ${OUTPUT_FILE} failed. ---"
    fi
done