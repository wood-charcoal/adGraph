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

#define DEBUG
#define PRINT_LIMIT 20

using namespace std;

// --- Start of Graph Generation Function (from page_rank_generator.cpp) ---
void generate_random_csc_graph(size_t n, size_t nnz,
                               std::vector<int> &destination_offsets_h,
                               std::vector<int> &source_indices_h,
                               std::vector<float> &weights_h)
{

    // Set a seed for reproducibility
    srand(42);

    if (nnz == 0)
    {
        destination_offsets_h.assign(n + 1, 0);
        return;
    }

    // Limit nnz to the max possible for a simple directed graph (no self-loops)
    size_t max_nnz = n * (n - 1);
    if (nnz > max_nnz)
    {
        nnz = max_nnz;
    }

    // Edges will be stored as pairs: (destination, source)
    std::vector<std::pair<int, int>> edges;
    std::map<int, int> out_degree_map;
    std::set<std::pair<int, int>> edge_set;

    // Generate edges until we have nnz unique edges
    while (edges.size() < nnz)
    {
        int src = rand() % n;
        int dst = rand() % n;

        if (src != dst)
        {
            if (edge_set.find({src, dst}) == edge_set.end())
            {
                edges.push_back({dst, src});
                edge_set.insert({src, dst});
                out_degree_map[src]++;
            }
        }
    }
    nnz = edges.size(); // Update nnz based on the actual number of generated unique edges

    // Sort edges by destination (column)
    std::sort(edges.begin(), edges.end());

    destination_offsets_h.resize(n + 1, 0);
    source_indices_h.resize(nnz);
    weights_h.resize(nnz);

    int current_nnz_count = 0;
    for (size_t i = 0; i < n; ++i)
    {
        destination_offsets_h[i] = current_nnz_count;

        for (const auto &edge : edges)
        {
            if (edge.first == (int)i)
            {
                int src = edge.second;
                source_indices_h[current_nnz_count] = src;

                // CRITICAL VALIDATION CHECK
                if (src >= (int)n)
                {
                    fprintf(stderr, "INTERNAL ERROR: Generated source index %d >= total vertices %zu\n", src, n);
                    exit(1);
                }

                // PageRank edge weight: 1 / out-degree of the source
                weights_h[current_nnz_count] = 1.0f / (out_degree_map.count(src) ? out_degree_map[src] : 1.0f);
                current_nnz_count++;
            }
        }
    }

    // Ensure current_nnz_count matches nnz
    if (current_nnz_count != (int)nnz)
    {
        fprintf(stderr, "INTERNAL ERROR: Generated edge count mismatch (%d vs %zu)\n", current_nnz_count, nnz);
        exit(1);
    }

    destination_offsets_h[n] = nnz;

#ifdef DEBUG
    printf("--- Generated Graph (n=%zu, nnz=%zu) ---\n", n, nnz);
    printf("destination_offsets_h (size %zu): ", destination_offsets_h.size());
    for (int val : destination_offsets_h)
        printf("%d ", val);
    printf("\n");
    printf("source_indices_h (size %zu): ", source_indices_h.size());
    for (int val : source_indices_h)
        printf("%d ", val);
    printf("\n");
    printf("weights_h (size %zu): ", weights_h.size());
    for (float val : weights_h)
        printf("%f ", val);
    printf("\n-----------------------------------------\n");
#endif
}
// --- End of Graph Generation Function ---

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
    // float weights_h[] = {0.333333f, 0.5f, 0.333333f, 0.5f, 0.5f, 1.0f, 0.333333f, 0.5f, 0.5f, 0.5f};
    // int destination_offsets_h[] = {0, 1, 3, 4, 6, 8, 10};
    // int source_indices_h[] = {2, 0, 2, 0, 4, 5, 2, 3, 3, 4};
    // float bookmark_h[] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    // --- USER DEFINED GRAPH PARAMETERS ---
    size_t n = 200;   // Number of vertices
    size_t nnz = 300; // Number of non-zero elements (edges)
    // -------------------------------------

    size_t vert_sets = 2, edge_sets = 1;
    float alpha1 = 0.9f;
    void *alpha1_p = (void *)&alpha1;

    // Containers for graph data
    std::vector<float> weights_v;
    std::vector<int> destination_offsets_v;
    std::vector<int> source_indices_v;

    // --- Graph Generation ---
    generate_random_csc_graph(n, nnz, destination_offsets_v, source_indices_v, weights_v);
    nnz = weights_v.size(); // Update nnz in case it was capped by the generator

    // Get raw pointers from vectors for C API
    float *weights_h = weights_v.data();
    int *destination_offsets_h = destination_offsets_v.data();
    int *source_indices_h = source_indices_v.data();

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

    // Initialize PageRank-specific data
    // Initial bookmark/personalization vector (can be initialized randomly or set to uniform 1.0/n)
    float *bookmark_h = (float *)calloc(n, sizeof(float));
    for (size_t i = 0; i < n; ++i)
    {
        bookmark_h[i] = 1.0f / (float)n; // Uniform personalization vector
    }

    vertex_dim[0] = (void *)bookmark_h;
    vertex_dim[1] = (void *)pr_1;
    vertex_dimT[0] = HIPBLAS_R_32F;
    vertex_dimT[1] = HIPBLAS_R_32F;

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
    NVGRAPH_CHECK(nvgraphSetVertexData(handle, graph, vertex_dim[0], 0));
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
#ifdef PRINT_LIMIT
    size_t print_limit = n < PRINT_LIMIT ? n : PRINT_LIMIT;
#else
    size_t print_limit = n;
#endif
    for (size_t i = 0; i < print_limit; i++)
        printf("Node %zu: %f\n", i, pr_1[i]);
    if (n > print_limit) printf("...\n");
    printf("\n");

    // Cleanup
    NVGRAPH_CHECK(nvgraphDestroyGraphDescr(handle, graph));
    NVGRAPH_CHECK(nvgraphDestroy(handle));
    free(pr_1);
    free(bookmark_h);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);
    return 0;
}