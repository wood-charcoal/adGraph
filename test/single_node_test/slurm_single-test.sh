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

NVGRAPH_INSTALL_ROOT="$HOME/projects/testing/adGraph/dist"
NVGRAPH_LIB_DIR="$NVGRAPH_INSTALL_ROOT/lib64" 
NVGRAPH_INCLUDE_DIR="$NVGRAPH_INSTALL_ROOT/include/nvgraph"
LAPACK_INSTALL_ROOT="$HOME/projects/lapack-3.12.0/install"

DEFAULT_ROCM_PATH="/public/software/compiler/rocm/dtk-24.04"

export DTK_ROOT="$DEFAULT_ROCM_PATH"
export PATH="$DTK_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="${LAPACK_INSTALL_ROOT}/lib64:${NVGRAPH_LIB_DIR}:$DTK_ROOT/lib:/public/software/compiler/gcc-8.2.0/isl-0.18/lib:$LD_LIBRARY_PATH"
export HIP_PATH="$DTK_ROOT"

export CC="$DTK_ROOT/llvm/bin/clang"
export CXX="$DTK_ROOT/bin/hipcc"
export HIPCC="$DTK_ROOT/bin/hipcc"

# ---------------------- Define files list ----------------------

FILES_LIST=(
    "page_rank.cpp"
    "page_rank_rand_graph.cpp"
    "triangle_count.cpp"
    "triangle_count_rand_graph.cpp"
)

if [ ${#FILES_LIST[@]} -eq 0 ]; then
    echo "Error: FILES_LIST is empty."
    exit 1
fi

echo "Processing the following files:"
printf ' - %s\n' "${FILES_LIST[@]}"

# ---------------------- compile scripts ----------------------

COMPILED_OUTPUTS=()

echo "Starting compilation..."
# Change directory to the source code location before compiling
cd ${TEST_CODE_DIR}

for TEST_FILE_NAME in "${FILES_LIST[@]}"; do
    
    # Check if the file exists before attempting to compile
    if [ ! -f "${TEST_FILE_NAME}" ]; then
        echo "Warning: Source file not found: ${TEST_FILE_NAME}. Skipping compilation."
        continue
    fi
    
    # Remove the .cpp extension and add .out
    OUTPUT_NAME="${TEST_FILE_NAME%.cpp}.out"

    echo "Compiling ${TEST_FILE_NAME} to ${OUTPUT_NAME}..."
    
    # Note: Compilation is run from within $TEST_CODE_DIR, so we use just the filename.
    hipcc -std=c++14 "${TEST_FILE_NAME}" -g -o "${OUTPUT_NAME}" \
        -I${NVGRAPH_INCLUDE_DIR} \
        -I${LAPACK_INSTALL_ROOT}/include \
        -I${DTK_ROOT}/include \
        -I${DTK_ROOT}/include/hip \
        -L${NVGRAPH_LIB_DIR} \
        -L${LAPACK_INSTALL_ROOT}/lib64 \
        -lnvgraph \
        -lcblas -lblas -llapack \
        -lhipblas -lhipsparse -lhiprand -lhipsolver -lrocsparse -lrocrand -lrocsolver -lamdhip64 -lpthread
        
    if [ $? -eq 0 ]; then
        echo "${OUTPUT_NAME} compiled successfully."
        COMPILED_OUTPUTS+=("${OUTPUT_NAME}")
    else
        echo "Compilation failed for ${TEST_FILE_NAME}. Exiting."
        # If compilation fails, change back to submission directory and exit
        cd ${SLURM_SUBMIT_DIR}
        exit 1 
    fi
done

# Change back to the submission directory to run the executables
cd ${SLURM_SUBMIT_DIR}
echo "Compilation completed. Executables are in ${TEST_CODE_DIR}."

echo "--------------------------------------------------------"

# ---------------------- run tests ----------------------

echo "Running tests..."

# The executables are located in $TEST_CODE_DIR
RUN_DIR=${TEST_CODE_DIR}

if [ ${#COMPILED_OUTPUTS[@]} -eq 0 ]; then
    echo "No executables were successfully compiled to run."
    exit 0
fi

for OUTPUT_FILE in "${COMPILED_OUTPUTS[@]}"; do
    echo "--- Running ${OUTPUT_FILE} ---"
    # Execute the file using its full path
    ${RUN_DIR}/${OUTPUT_FILE}
    if [ $? -eq 0 ]; then
        echo "--- Test ${OUTPUT_FILE} passed. ---"
    else
        echo "--- Test ${OUTPUT_FILE} failed. ---"
    fi
done
