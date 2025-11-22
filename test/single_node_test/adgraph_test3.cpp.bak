#include "stdlib.h"
#include "inttypes.h"
#include "stdio.h"
#include "nvgraph.h"
#include <hipblas.h>
#include <iostream>
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

    // Example of graph (CSR format)
    const size_t n = 7, nnz = 12, vertex_numsets = 2, edge_numset = 0;
    int source_offsets_h[] = {0, 1, 3, 4, 6, 8, 10, 12};
    int destination_indices_h[] = {5, 0, 2, 0, 4, 5, 2, 3, 3, 4, 1, 5};

    // where to store results (distances from source) and where to store results (predecessors in search tree)
    int bfs_distances_h[n], bfs_predecessors_h[n];

    // nvgraph variables
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSRTopology32I_t CSR_input;
    hipblasDatatype_t *vertex_dimT;
    size_t distances_index = 0;
    size_t predecessors_index = 1;
    vertex_dimT = (hipblasDatatype_t *)malloc(vertex_numsets * sizeof(hipblasDatatype_t));
    vertex_dimT[distances_index] = HIPBLAS_R_32I;
    vertex_dimT[predecessors_index] = HIPBLAS_R_32I;

    // Creating nvgraph objects
    NVGRAPH_CHECK(nvgraphCreate(&handle));
    NVGRAPH_CHECK(nvgraphCreateGraphDescr(handle, &graph));

    // Set graph connectivity and properties (tranfers)
    CSR_input = (nvgraphCSRTopology32I_t)malloc(sizeof(struct nvgraphCSCTopology32I_st));
    CSR_input->nvertices = n;
    CSR_input->nedges = nnz;
    CSR_input->source_offsets = source_offsets_h;
    CSR_input->destination_indices = destination_indices_h;
    NVGRAPH_CHECK(nvgraphSetGraphStructure(handle, graph, (void *)CSR_input, NVGRAPH_CSR_32));
    NVGRAPH_CHECK(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    int source_vert = 1;

    // Setting the traversal parameters
    nvgraphTraversalParameter_t traversal_param;
    nvgraphTraversalParameterInit(&traversal_param);
    nvgraphTraversalSetDistancesIndex(&traversal_param, distances_index);
    nvgraphTraversalSetPredecessorsIndex(&traversal_param, predecessors_index);
    nvgraphTraversalSetUndirectedFlag(&traversal_param, false);

    // Computing traversal using BFS algorithm
    NVGRAPH_CHECK(nvgraphTraversal(handle, graph, NVGRAPH_TRAVERSAL_BFS, &source_vert, traversal_param));

    // Get result
    NVGRAPH_CHECK(nvgraphGetVertexData(handle, graph, (void *)bfs_distances_h, distances_index));
    NVGRAPH_CHECK(nvgraphGetVertexData(handle, graph, (void *)bfs_predecessors_h, predecessors_index));

    // expect bfs distances_h = (1 0 1 3 3 2 2147483647)
    for (int i = 0; i < n; i++)
        printf("Distance to vertex %d: %i\n", i, bfs_distances_h[i]);
    printf("\n");
    // expect bfs predecessors = (1 -1 1 5 5 0 -1)
    for (int i = 0; i < n; i++)
        printf("Predecessor of vertex %d: %i\n", i, bfs_predecessors_h[i]);
    printf("\n");
    free(vertex_dimT);
    free(CSR_input);
    NVGRAPH_CHECK(nvgraphDestroyGraphDescr(handle, graph));
    NVGRAPH_CHECK(nvgraphDestroy(handle));
    return 0;
}