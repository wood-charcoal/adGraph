#include "stdlib.h"
#include "inttypes.h"
#include "stdio.h"
#include "nvgraph.h"
#include <hipblas.h>
#include <ctime>
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

    size_t n = 5, nnz = 10, vertex_numsets = 2, edge_numsets = 1;
    float alpha = 1.0, beta = 0.0;
    void *alpha_p = (void *)&alpha, *beta_p = (void *)&beta;
    void **vertex_dim;
    hipblasDatatype_t edge_dimT = HIPBLAS_R_32F;
    hipblasDatatype_t *vertex_dimT;

    // nvgraph variables
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSRTopology32I_t CSR_input;

    // Init host data
    vertex_dim = (void **)malloc(vertex_numsets * sizeof(void *));
    vertex_dimT = (hipblasDatatype_t *)malloc(vertex_numsets * sizeof(hipblasDatatype_t));
    CSR_input = (nvgraphCSRTopology32I_t)malloc(sizeof(struct nvgraphCSRTopology32I_st));
    float x_h[] = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
    float y_h[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    vertex_dim[0] = (void *)x_h;
    vertex_dim[1] = (void *)y_h;
    vertex_dimT[0] = HIPBLAS_R_32F;
    vertex_dimT[1] = HIPBLAS_R_32F;
    float weights_h[] = {1.0f, 4.0f, 2.0f, 3.0f, 5.0f, 7.0f, 8.0f, 9.0f, 6.0f, 1.5f};
    int source_offsets_h[] = {0, 2, 4, 7, 9, 10};
    int destination_indices_h[] = {0, 1, 1, 2, 0, 3, 4, 2, 4, 2};
    NVGRAPH_CHECK(nvgraphCreate(&handle));
    NVGRAPH_CHECK(nvgraphCreateGraphDescr(handle, &graph));
    CSR_input->nvertices = n;
    CSR_input->nedges = nnz;
    CSR_input->source_offsets = source_offsets_h;
    CSR_input->destination_indices = destination_indices_h;

    // Set graph connectivity and properties (tranfers)
    NVGRAPH_CHECK(nvgraphSetGraphStructure(handle, graph, (void *)CSR_input, NVGRAPH_CSR_32));
    NVGRAPH_CHECK(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    for (int i = 0; i < vertex_numsets; ++i)
        NVGRAPH_CHECK(nvgraphSetVertexData(handle, graph, vertex_dim[i], i));
    NVGRAPH_CHECK(nvgraphAllocateEdgeData(handle, graph, edge_numsets, &edge_dimT));
    NVGRAPH_CHECK(nvgraphSetEdgeData(handle, graph, (void *)weights_h, 0));

    // Solve
    NVGRAPH_CHECK(nvgraphSrSpmv(handle, graph, 0, alpha_p, 0, beta_p, 1, NVGRAPH_PLUS_TIMES_SR));

    // Get result
    NVGRAPH_CHECK(nvgraphGetVertexData(handle, graph, (void *)y_h, 1));
    for (int i = 0; i < n; i++)
        printf("%f ", y_h[i]);
    printf("\n");

    // Clean
    NVGRAPH_CHECK(nvgraphDestroyGraphDescr(handle, graph));
    NVGRAPH_CHECK(nvgraphDestroy(handle));
    free(vertex_dim);
    free(vertex_dimT);
    free(CSR_input);
    return 0;
}