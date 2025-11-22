#include "hip/hip_runtime.h"
#include <iostream>
#include <vector>
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <cmath> // For std::fabs

// --- HIP Error Checking Helper ---
// A robust error macro is essential for debugging HIP applications.
#define HIP_CHECK(command) \
    do { \
        hipError_t err = command; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP Error: %s (%d) at %s:%d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// --- HIP Kernel: Vector Addition ---
// The __global__ specifier marks this function to be run on the GPU device.
__global__ void vectorAdd(float* C, const float* A, const float* B, int N) {
    // Calculate global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (i < N) {
        // Simple operation: C = A + B + 1.0
        // The addition of '1.0f' helps ensure the computation occurs.
        C[i] = A[i] + B[i] + 1.0f;
    }
}

// --- Main Application ---
int main(int argc, char** argv) {
    // 1. MPI Initialization
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "ERROR: This test requires at least 2 MPI ranks.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const int N = 1024 * 1024; // 1 Million elements
    const size_t bytes = N * sizeof(float);
    
    printf("MPI Rank %d starting execution. Array size: %d elements (%.2f MB)\n", rank, N, bytes / (1024.0 * 1024.0));

    // 2. HIP Device Assignment
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    
    // Cyclic distribution: Assign a GPU device to each MPI rank
    int device_id = rank % device_count;
    HIP_CHECK(hipSetDevice(device_id));
    if (device_count != 1) {
        fprintf(stderr, "WARNING: Rank %d expected 1 visible device but found %d.\n", rank, device_count);
    }
    printf("Rank %d assigned to HIP Device %d out of %d total devices.\n", rank, device_id, device_count);

    // --- Host and Device Buffers ---
    std::vector<float> h_A(N), h_B(N), h_C(N); // Host buffers
    float *d_A, *d_B, *d_C;                  // Device pointers

    // Allocate Device Memory
    HIP_CHECK(hipMalloc((void**)&d_A, bytes));
    HIP_CHECK(hipMalloc((void**)&d_B, bytes));
    HIP_CHECK(hipMalloc((void**)&d_C, bytes));

    // 3. Synchronization
    printf("Rank %d has allocated memory. Synchronizing...\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Rank 0: Sender and Initializer ---
    if (rank == 0) {
        printf("Rank 0: Initializing and sending data to Rank 1...\n");
        
        // Initialize Host Data
        for (int i = 0; i < N; ++i) {
            h_A[i] = (float)i;
            h_B[i] = (float)i * 2.0f;
        }

        // Copy Host -> Device (Rank 0's GPU)
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice));

        // Send Device Buffer A directly to Rank 1's device buffer A (GPU-Aware MPI)
        // This is the core test of MPI/HIP integration.
        printf("Rank 0: Sending device buffer d_A to Rank 1...\n");
        MPI_Send(d_A, N, MPI_FLOAT, 1, 100, MPI_COMM_WORLD);
        printf("Rank 0: Data sent.\n");

        // We only send A for simplicity; Rank 1 will initialize B itself in this minimal test.
        // For a full application, Rank 0 would also send d_B.
    } 
    
    // --- Rank 1: Receiver, Compute, and Validator ---
    else if (rank == 1) {
        // Initialize Host Data for B (only needed for calculation and copy)
        for (int i = 0; i < N; ++i) {
            h_B[i] = (float)i * 2.0f;
        }
        // Copy Host -> Device B (Rank 1's GPU)
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice));

        // Receive Data directly into Device Buffer A (GPU-Aware MPI)
        printf("Rank 1: Waiting to receive data into device buffer d_A...\n");
        MPI_Status status;
        MPI_Recv(d_A, N, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &status);
        printf("Rank 1: Data received. Launching HIP kernel...\n");

        // Kernel Launch Parameters
        const int blockSize = 256;
        const int gridSize = (N + blockSize - 1) / blockSize;

        // Launch the Kernel
        hipLaunchKernelGGL(vectorAdd, dim3(gridSize), dim3(blockSize), 0, 0, d_C, d_A, d_B, N);
        
        // Wait for the GPU to complete its work
        HIP_CHECK(hipDeviceSynchronize());

        // Copy Result Device -> Host
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, bytes, hipMemcpyDeviceToHost));
        printf("Rank 1: Results copied back to host. Validating...\n");

        // Validation
        int errors = 0;
        for (int i = 0; i < N; ++i) {
            float expected = (float)i + (float)i * 2.0f + 1.0f;
            if (std::fabs(h_C[i] - expected) > 1e-5) {
                if (errors < 10) { // Print only the first 10 errors
                    fprintf(stderr, "Error at index %d: Expected %f, Got %f\n", i, expected, h_C[i]);
                }
                errors++;
            }
        }

        if (errors == 0) {
            printf("Rank 1: SUCCESS! Vector addition and GPU-Aware MPI transfer validated.\n");
        } else {
            printf("Rank 1: FAILURE! Found %d errors.\n", errors);
        }
    }

    // 4. Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    // 5. MPI Finalization
    MPI_Finalize();
    return 0;
}