#!/bin/bash
#SBATCH -J compile
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


# ========================== Set HIP/ROCm Environment ==========================
# Default ROCm installation path - modify if needed
DEFAULT_ROCM_PATH="/public/software/compiler/rocm/dtk-24.04"

# Check if DTK_ROOT is already set
if [ -z "$DTK_ROOT" ]; then
    if [ -d "$DEFAULT_ROCM_PATH" ]; then
        export DTK_ROOT="$DEFAULT_ROCM_PATH"
        echo "Set DTK_ROOT to default path: $DTK_ROOT"
    else
        echo "Error: DTK_ROOT not set and default ROCm path not found at $DEFAULT_ROCM_PATH"
        echo "Please set DTK_ROOT to your ROCm installation path"
        echo "Example: export DTK_ROOT=/public/software/compiler/rocm/dtk-24.04"
        exit 1
    fi
else
    echo "Using existing DTK_ROOT: $DTK_ROOT"
fi

# Verify ROCm installation
if [ ! -d "$DTK_ROOT" ]; then
    echo "Error: ROCm installation not found at $DTK_ROOT"
    exit 1
fi

# Set up environment variables
export PATH=$DTK_ROOT/hip/bin/hipify:/public/software/compiler/gcc-8.2.0/bin:$DTK_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$DTK_ROOT/lib:/public/software/compiler/gcc-8.2.0/isl-0.18/lib:$LD_LIBRARY_PATH
export HIP_PATH=$DTK_ROOT
export LAPACK_INSTALL_ROOT=$HOME/projects/lapack-3.12.0/install

# Set compiler paths
export CC="$DTK_ROOT/llvm/bin/clang"
export CXX="$DTK_ROOT/bin/hipcc"
export HIPCC="$DTK_ROOT/bin/hipcc"

# Build configuration
export BUILD_TYPE=Release
export BUILD_ABI=ON

echo "HIP/ROCm environment configured:"
echo "  DTK_ROOT: $DTK_ROOT"
echo "  HIP_PATH: $HIP_PATH"
echo "  CC: $CC"
echo "  CXX: $CXX"
echo "  HIPCC: $HIPCC"
echo "  BUILD_TYPE: $BUILD_TYPE"

# Check if required tools are available
echo ""
echo "Checking for required tools:"

if command -v hipcc &> /dev/null; then
    echo "  ✓ hipcc found: $(which hipcc)"
    hipcc --version
else
    echo "  ✗ hipcc not found in PATH"
fi

echo ""
echo "Environment setup complete!"

# ========================== Build Project ==========================
echo "Starting build process..."
./build.sh -v

date