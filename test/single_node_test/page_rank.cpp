/*
    PageRank example using nvgraph API
*/
#include "stdlib.h"
#include "inttypes.h"
#include "stdio.h"
#include "nvgraph.h"
#include <hipblas.h>
#include <ctime>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <map>
#include <set>

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
    size_t n = 6, nnz = 10, vert_sets = 2, edge_sets = 1;
    float alpha1 = 0.9f;
    void *alpha1_p = (void *)&alpha1;

    // nvgraph variables
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    hipblasDatatype_t edge_dimT = HIPBLAS_R_32F;
    hipblasDatatype_t *vertex_dimT;

    // Allocate host data
    float *pr_1 = (float *)malloc(n * sizeof(float));
    void **vertex_dim = (void **)malloc(vert_sets * sizeof(void *));
    vertex_dimT = (hipblasDatatype_t *)malloc(vert_sets * sizeof(hipblasDatatype_t));
    CSC_input = (nvgraphCSCTopology32I_t)malloc(sizeof(struct nvgraphCSCTopology32I_st));

    float weights_h[] = {0.333333f, 0.5f, 0.333333f, 0.5f, 0.5f, 1.0f, 0.333333f, 0.5f, 0.5f, 0.5f};
    int destination_offsets_h[] = {0, 1, 3, 4, 6, 8, 10};
    int source_indices_h[] = {2, 0, 2, 0, 4, 5, 2, 3, 3, 4};
    float bookmark_h[] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    vertex_dim[0] = (void *)bookmark_h;
    vertex_dim[1] = (void *)pr_1;
    vertex_dimT[0] = HIPBLAS_R_32F;
    vertex_dimT[1] = HIPBLAS_R_32F, vertex_dimT[2] = HIPBLAS_R_32F;

    // Starting nvgraph
    NVGRAPH_CHECK(nvgraphCreate(&handle));
    NVGRAPH_CHECK(nvgraphCreateGraphDescr(handle, &graph));
    CSC_input->nvertices = n;
    CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;

    // Set graph connectivity and properties (tranfers)
    NVGRAPH_CHECK(nvgraphSetGraphStructure(handle, graph, (void *)CSC_input, NVGRAPH_CSC_32));
    NVGRAPH_CHECK(nvgraphAllocateVertexData(handle, graph, vert_sets, vertex_dimT));
    NVGRAPH_CHECK(nvgraphAllocateEdgeData(handle, graph, edge_sets, &edge_dimT));

    // Set bookmark vector (vertex property 0) and allocate for result (vertex property 1)
    for (int i = 0; i < 2; ++i)
        NVGRAPH_CHECK(nvgraphSetVertexData(handle, graph, vertex_dim[i], i));
    // Set edge weights (edge property 0)
    NVGRAPH_CHECK(nvgraphSetEdgeData(handle, graph, (void *)weights_h, 0));

    // Time counting
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Run PageRank:
    // Pagerank(handle, graph, edge_set_index, alpha_p, personalization_vector_index,
    //          has_guess, result_vector_index, tolerance, max_iter)
    NVGRAPH_CHECK(nvgraphPagerank(handle, graph, 0, alpha1_p, 0, 0, 1, 0.0f, 0));

    clock_gettime(CLOCK_MONOTONIC, &end);
    long diff_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    printf("\n--- PageRank execution time ---\n");
    printf("Time: %ld ns\n\n", diff_ns);

    // Get result
    NVGRAPH_CHECK(nvgraphGetVertexData(handle, graph, vertex_dim[1], 1));
    printf("\n--- PageRank Result ---\n");
    printf("pr_1, alpha = 0.9\n");
    for (size_t i = 0; i < n; i++)
        printf("Node %zu: %f\n", i, pr_1[i]);
    printf("\n");

    // Cleanup
    NVGRAPH_CHECK(nvgraphDestroyGraphDescr(handle, graph));
    NVGRAPH_CHECK(nvgraphDestroy(handle));
    free(pr_1);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);
    return 0;
}