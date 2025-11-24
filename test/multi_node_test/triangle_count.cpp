/*
    Triangle counting using the nvgraph API with Multi-Node/MPI support
*/
#include "stdlib.h"
#include "inttypes.h"
#include "stdio.h"
#include "nvgraph.h"
#include <hipblas.h>
#include <ctime>
#include <iostream>
#include <time.h>
#include <vector>
#include <algorithm>
#include <random>
#include <set>
#include <mpi.h>

#define DEBUG
// #define PRINT_LIMIT 20

using namespace std;

void check(nvgraphStatus_t status, const char *file, int line)
{
    if (status != NVGRAPH_STATUS_SUCCESS)
    {
        printf("NVGRAPH ERROR: %d at %s:%d\n", status, file, line);
        exit(0);
    }
}

#define NVGRAPH_CHECK(status) check(status, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    // --- MPI Initialization ---
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // --------------------------

    // nvgraph variables
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSRTopology32I_t CSR_input;

    // Initialize host data structure for nvGraph
    CSR_input = (nvgraphCSRTopology32I_t)malloc(sizeof(struct
                                                       nvgraphCSRTopology32I_st));

    // Undirected graph:
    // 0       2-------4
    //  \     / \     / \
    //   \   /   \   /   \
    //    \ /     \ /     \
    //     1-------3-------5
    // 3 triangles
    // CSR of lower triangular of adjacency matrix:
    const size_t n = 6, nnz = 8;
    int source_offsets[] = {0, 0, 1, 2, 4, 6, 8};
    int destination_indices[] = {0, 1, 1, 2, 2, 3, 3, 4};

    // Start nvgraph (Initiates distributed context using MPI_COMM_WORLD)
    NVGRAPH_CHECK(nvgraphCreate(&handle));
    NVGRAPH_CHECK(nvgraphCreateGraphDescr(handle, &graph));

    // Transfer generated data to nvGraph structure
    CSR_input->nvertices = n;
    CSR_input->nedges = nnz;
    CSR_input->source_offsets = source_offsets;
    CSR_input->destination_indices = destination_indices;

    // Set graph structure (This is where the library partitions the graph across ranks)
    NVGRAPH_CHECK(nvgraphSetGraphStructure(handle, graph, (void *)CSR_input, NVGRAPH_CSR_32));

    uint64_t trcount = 0;

    // Time counting (Only on root rank)
    struct timespec start, end;
    if (rank == 0)
    {
        printf("--- Starting Multi-Node Triangle Counting ---\n");
        printf("Total Ranks: %d, Vertices: %zu, Edges in CSR: %zu\n", size, n, nnz);
        clock_gettime(CLOCK_MONOTONIC, &start);
    }

    // Execute triangle counting (Collective call executed by ALL ranks)
    NVGRAPH_CHECK(nvgraphTriangleCount(handle, graph, &trcount));

    // Report results (Only on root rank)
    if (rank == 0)
    {
        clock_gettime(CLOCK_MONOTONIC, &end);
        long diff_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
        printf("\n--- Triangle Counting execution time ---\nTime: %ld ns\n", diff_ns);

        // The result 'trcount' on Rank 0 now holds the final, global triangle count.
        printf("\nTotal Undirected Triangles count: %" PRIu64 "\n", trcount);
    }

    // Cleanup (Executed by ALL ranks)
    free(CSR_input);
    NVGRAPH_CHECK(nvgraphDestroyGraphDescr(handle, graph));
    NVGRAPH_CHECK(nvgraphDestroy(handle));

    // --- MPI Finalization ---
    MPI_Finalize();
    // ------------------------

    return 0;
}