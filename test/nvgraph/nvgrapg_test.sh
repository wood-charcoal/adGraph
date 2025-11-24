#!/bin/bash
# CUDA 10.2 nvgraph Test Script

# Define the list of source files (INCLUDING the .cpp suffix)
FILES_LIST=(
    "page_rank.cpp"
    "page_rank_rand_graph.cpp"
    "triangle_count.cpp"
    # Add other filenames here with the .cpp extension
)

# Array to store the names of successfully compiled executables (e.g., page_rank.out)
COMPILED_OUTPUTS=()

# --- Compilation Phase ---
echo "--- Starting Compilation Phase ---"

for SOURCE_FILE in "${FILES_LIST[@]}"; do
    
    # 1. Check if the source file exists
    if [ ! -f "${SOURCE_FILE}" ]; then
        echo "Error: Source file not found: ${SOURCE_FILE}. Skipping."
        continue
    fi
    
    # 2. Derive the output filename by replacing .cpp with .out
    # We use basename to ensure it works even if the list contained paths.
    BASE_NAME=$(basename "${SOURCE_FILE}")
    OUTPUT_NAME="${BASE_NAME%.cpp}.out"

    echo "=> Compiling ${SOURCE_FILE} to ${OUTPUT_NAME}"
    
    # Compilation Command
    nvcc "${SOURCE_FILE}" -o "${OUTPUT_NAME}" -lnvgraph -lcudart -lstdc++
    
    if [ $? -eq 0 ]; then
        echo "Finished Compiling ${OUTPUT_NAME} successfully."
        COMPILED_OUTPUTS+=("${OUTPUT_NAME}")
    else
        echo "ERROR: Compilation failed for ${SOURCE_FILE}. Exiting."
        exit 1
    fi
done

echo "--- Finished Compilation Phase ---"

# --- Execution Phase ---
echo "--- Starting Execution Phase ---"

if [ ${#COMPILED_OUTPUTS[@]} -eq 0 ]; then
    echo "No executables were successfully compiled. Skipping execution."
    exit 0
fi

for OUTPUT_FILE in "${COMPILED_OUTPUTS[@]}"; do
    echo ""
    echo "=> Executing ${OUTPUT_FILE}"
    ./"${OUTPUT_FILE}"
    
    if [ $? -eq 0 ]; then
        echo "Finished ${OUTPUT_FILE} successfully."
    else
        echo "ERROR: ${OUTPUT_FILE} failed during execution."
    fi
done

echo "--- Finished Execution Phase ---"