#!/bin/bash
#SBATCH -J test_multi
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
TEST_CODE_DIR="$HOME/projects/testing/adGraph/test/multi_node_test" 

NVGRAPH_INSTALL_ROOT="$HOME/projects/adGraph/dist"
NVGRAPH_LIB_DIR="$NVGRAPH_INSTALL_ROOT/lib64"
NVGRAPH_INCLUDE_DIR="$NVGRAPH_INSTALL_ROOT/include/nvgraph"
LAPACK_INSTALL_ROOT="$HOME/projects/lapack-3.12.0/install"

DEFAULT_ROCM_PATH="/public/software/compiler/rocm/dtk-24.04"

export DTK_ROOT="$DEFAULT_ROCM_PATH"
export PATH="$DTK_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="${LAPACK_INSTALL_ROOT}/lib64:${NVGRAPH_LIB_DIR}:/public/software/compiler/gcc-8.2.0/isl-0.18/lib:$DTK_ROOT/lib:$LD_LIBRARY_PATH"
export HIP_PATH="$DTK_ROOT"

export CC="$DTK_ROOT/llvm/bin/clang"
export CXX="$DTK_ROOT/bin/hipcc"
export HIPCC="$DTK_ROOT/bin/hipcc"
export MPICH_GPU_SUPPORT_ENABLED=1
export HSA_ENABLE_SDMA=0 

# ---------------------- Define files list ----------------------

FILES_LIST=(
    "page_rank.cpp"
    "page_rank_rand_graph.cpp"
    "triangle_count.cpp"
    "triangle_count_rand_graph.cpp"
)

if [ ${#FILES_LIST[@]} -eq 0 ]; then
    echo "Error: FILES_LIST is empty. Exiting."
    exit 1
fi

echo "Processing the following files:"
printf ' - %s\n' "${FILES_LIST[@]}"

# Array to store the names of successfully compiled executables
COMPILED_OUTPUTS=()

# ---------------------- compile scripts ----------------------

MPI_LINK_FLAGS=$(mpicxx -showme:link)

echo "Starting compilation..."
# Change directory to the source code location before compiling
cd ${TEST_CODE_DIR}

for SOURCE_FILE in "${FILES_LIST[@]}"; do
    
    # Check if the file exists before attempting to compile
    if [ ! -f "${SOURCE_FILE}" ]; then
        echo "Warning: Source file not found: ${SOURCE_FILE}. Skipping compilation."
        continue
    fi
    
    # Derive the output filename by replacing .cpp with .out
    OUTPUT_NAME="${SOURCE_FILE%.cpp}.out"

    echo "Compiling ${SOURCE_FILE} to ${OUTPUT_NAME}..."
    
    hipcc -std=c++14 "${SOURCE_FILE}" -g -o "${OUTPUT_NAME}" \
        -I${NVGRAPH_INCLUDE_DIR} \
        -I${LAPACK_INSTALL_ROOT}/include \
        -I${DTK_ROOT}/include \
        -I${DTK_ROOT}/include/hip \
        -L${NVGRAPH_LIB_DIR} \
        -L${LAPACK_INSTALL_ROOT}/lib64 \
        -lnvgraph \
        -lcblas -lblas -llapack \
        -lhipblas -lhipsparse -lhiprand -lhipsolver -lrocsparse -lrocrand -lrocsolver -lamdhip64 \
        -lpthread \
        $MPI_LINK_FLAGS
        
    if [ $? -eq 0 ]; then
        echo "${OUTPUT_NAME} compiled successfully."
        COMPILED_OUTPUTS+=("${OUTPUT_NAME}")
    else
        echo "Compilation failed for ${SOURCE_FILE}. Exiting."
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

# This variable is often used to ensure each rank targets a specific GPU on the node
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

echo "Running tests..."

# The executables are located in $TEST_CODE_DIR
RUN_DIR=${TEST_CODE_DIR}

if [ ${#COMPILED_OUTPUTS[@]} -eq 0 ]; then
    echo "No executables were successfully compiled to run."
    exit 0
fi

# Iterate over the list of successfully compiled executables
for OUTPUT_FILE in "${COMPILED_OUTPUTS[@]}"; do
    echo "--- Running ${OUTPUT_FILE} with mpirun ---"
    
    # Execute the file using its full path with mpirun
    # Note: mpirun automatically uses the SLURM allocation (2 tasks across 2 nodes).
    mpirun ${RUN_DIR}/${OUTPUT_FILE}
    
    if [ $? -eq 0 ]; then
        echo "--- Test ${OUTPUT_FILE} passed. ---"
    else
        echo "--- Test ${OUTPUT_FILE} failed. ---"
    fi
done