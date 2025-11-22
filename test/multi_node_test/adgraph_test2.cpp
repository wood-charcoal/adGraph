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

/**
 * @brief Generates a random undirected graph in CSR format.
 * All ranks will execute this to load the full graph.
 */
void generate_random_graph_csr(int n, int &nnz, int num_triangle,
                               std::vector<int> &source_offsets_h,
                               std::vector<int> &destination_indices_h)
{
    // Clear output vectors before generation
    source_offsets_h.clear();
    destination_indices_h.clear();

    // Constraint check
    if (n < 3 || nnz < 0)
    {
        std::cerr << "Error: Invalid parameters for graph generation. Need n >= 3 and nnz >= 0." << std::endl;
        nnz = 0;
        return;
    }

    // 1. Initialize random number generator with the fixed seed
    const unsigned SPECIFIC_SEED = 42;
    std::mt19937 generator(SPECIFIC_SEED);
    // Distribution for selecting any vertex from 0 to n-1
    std::uniform_int_distribution<int> distrib_v(0, n - 1);

    // Use a set to store (source, destination) pairs for the full symmetric graph
    std::set<std::pair<int, int>> full_edges;

    // Set to store the *sorted* vertex list of each guaranteed triangle to ensure uniqueness.
    std::set<std::vector<int>> guaranteed_triangles;

    // --- RANDOMIZED, OVERLAPPING TRIANGLE GENERATION ---

    int current_triangles = 0;
    int max_attempts = 1000 * num_triangle; // Safeguard against infinite loops
    int attempt = 0;

    // 2. Construct core symmetric triangles, allowing them to share vertices/edges.
    while (current_triangles < num_triangle && attempt < max_attempts)
    {
        attempt++;

        // Select three vertices
        int v1 = distrib_v(generator);
        int v2 = distrib_v(generator);
        int v3 = distrib_v(generator);

        // Check for invalid triangle (two or three vertices are the same)
        if (v1 == v2 || v2 == v3 || v3 == v1)
        {
            continue;
        }

        // Create a canonical representation of the triangle: [min_v, mid_v, max_v]
        std::vector<int> triangle_verts = {v1, v2, v3};
        std::sort(triangle_verts.begin(), triangle_verts.end());

        // Check for uniqueness
        if (guaranteed_triangles.find(triangle_verts) == guaranteed_triangles.end())
        {
            // New unique triangle found!
            guaranteed_triangles.insert(triangle_verts);
            current_triangles++;

            // Extract the vertices from the sorted list
            v1 = triangle_verts[0];
            v2 = triangle_verts[1];
            v3 = triangle_verts[2];

            // Insert all 6 directed edges for the full symmetric graph.
            // The 'full_edges' set ensures that shared edges are only stored once.

            // v1 <-> v2
            full_edges.insert({v1, v2});
            full_edges.insert({v2, v1});

            // v2 <-> v3
            full_edges.insert({v2, v3});
            full_edges.insert({v3, v2});

            // v3 <-> v1
            full_edges.insert({v3, v1});
            full_edges.insert({v1, v3});

#ifdef DEBUG
            std::cout << "Generated triangle vertices: {" << v1 << ", " << v2 << ", " << v3 << "}\n";
#endif
        }
    }

    // In case we couldn't generate the requested amount
    if (current_triangles < num_triangle && attempt >= max_attempts)
    {
        std::cerr << "Warning: Could only generate " << current_triangles << " unique triangles after maximum attempts.\n";
    }

    // 3. Fill remaining edges randomly up to the target NNZ (for the full graph)
    // NOTE: If the edges from the triangles already satisfy nnz, no new edges are added.
    attempt = 0;
    const int MAX_ATTEMPTS_FILLER = 5 * (nnz > 0 ? nnz : 1);

    while (full_edges.size() < (size_t)nnz && attempt < MAX_ATTEMPTS_FILLER)
    {
        int u = distrib_v(generator);
        int v = distrib_v(generator);
        attempt++;

        if (u != v)
        {
            // Check only one direction, if not present, add both symmetric edges
            if (full_edges.find({u, v}) == full_edges.end())
            {
                full_edges.insert({u, v});
                full_edges.insert({v, u}); // Add symmetric edge
            }
        }
    }

    // 4. Update nnz to the final actual edge count
    int final_nedges = full_edges.size();
    nnz = final_nedges;

    // 5. Convert Full Symmetric Edges to CSR format

    // Convert set to vector for sorting
    std::vector<std::pair<int, int>> full_symmetric_edges(full_edges.begin(), full_edges.end());

    // Sort by source vertex first, then destination vertex
    std::sort(full_symmetric_edges.begin(), full_symmetric_edges.end(), [](const auto &a, const auto &b)
              {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second; });

    // Resize and reserve space for CSR arrays
    source_offsets_h.resize(n + 1, 0);
    destination_indices_h.reserve(final_nedges);

    int src_vertex = 0;
    int des_offset = 0;

    // Fill CSR arrays
    for (const auto &edge : full_symmetric_edges)
    {
        int src = edge.first;
        int dst = edge.second;

        while (src > src_vertex)
        {
            source_offsets_h[++src_vertex] = des_offset;
        }

        destination_indices_h.push_back(dst);
        des_offset++;
    }

    while (src_vertex < n)
    {
        source_offsets_h[++src_vertex] = des_offset;
    }

#ifdef DEBUG
    // Print debug output only on Rank 0
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        // Print final CSR arrays
        std::cout << "Final CSR Representation:\n";
        std::cout << "Source Offsets: ";
        for (int i = 0; i < source_offsets_h.size(); i++)
        {
            std::cout << source_offsets_h[i] << " ";
        }
        std::cout << "\nDestination Indices: ";
        for (int i = 0; i < destination_indices_h.size(); i++)
        {
            std::cout << destination_indices_h[i] << " ";
        }
#ifdef PRINT_LIMIT
        size_t print_limit = n < PRINT_LIMIT ? n : PRINT_LIMIT;
#else
        size_t print_limit = n;
#endif
        std::cout << "\nAdjacency List View:\n";
        for (int i = 0; i < print_limit; i++)
        {
            std::cout << i << " -> { ";
            for (int j = source_offsets_h[i]; j < source_offsets_h[i + 1]; j++)
            {
                std::cout << destination_indices_h[j] << " ";
            }
            std::cout << "}\n";
        }
        if (n > print_limit)
            std::cout << "...\n\n";
    }
#endif
}

int main(int argc, char **argv)
{
    // --- MPI Initialization ---
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // --------------------------

    // --- Define graph parameters and seed ---
    // Note: Graph parameters should be consistent across all ranks
    int n = 6;
    int nnz = 20;
    int num_triangle = 3;

    // Host vectors for generated CSR data
    std::vector<int> h_source_offsets;
    std::vector<int> h_destination_indices;

    // Generate the random graph (Executed by ALL ranks)
    generate_random_graph_csr(n, nnz, num_triangle,
                              h_source_offsets, h_destination_indices);

    if (h_source_offsets.empty() || h_destination_indices.empty())
    {
        if (rank == 0)
            printf("Error: Graph generation failed.\n");
        MPI_Finalize();
        return 1; // Exit on error
    }

    // nvgraph variables
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSRTopology32I_t CSR_input;

    // Initialize host data structure for nvGraph
    CSR_input = (nvgraphCSRTopology32I_t)malloc(sizeof(struct
                                                       nvgraphCSRTopology32I_st));

    // Transfer generated data to nvGraph structure
    CSR_input->nvertices = n;
    CSR_input->nedges = nnz;
    CSR_input->source_offsets = h_source_offsets.data();
    CSR_input->destination_indices = h_destination_indices.data();

    // Start nvgraph (Initiates distributed context using MPI_COMM_WORLD)
    NVGRAPH_CHECK(nvgraphCreate(&handle));
    NVGRAPH_CHECK(nvgraphCreateGraphDescr(handle, &graph));

    // Set graph structure (This is where the library partitions the graph across ranks)
    NVGRAPH_CHECK(nvgraphSetGraphStructure(handle, graph, (void *)CSR_input, NVGRAPH_CSR_32));

    uint64_t trcount = 0;

    // Time counting (Only on root rank)
    struct timespec start, end;
    if (rank == 0)
    {
        printf("--- Starting Multi-Node Triangle Counting ---\n");
        printf("Total Ranks: %d, Vertices: %d, Edges in CSR: %d\n", size, n, nnz);
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
        printf("\nTotal Undirected Triangles count: %" PRIu64 " (Expected >= %d)\n", trcount, num_triangle);
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