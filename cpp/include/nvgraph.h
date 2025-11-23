/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _NVGRAPH_H_
#define _NVGRAPH_H_

#include "stddef.h"
#include "stdint.h"

#include "library_types.h"

#define NVG_HIP_TRY(T)                          \
	{                                           \
		if (T != hipSuccess)                    \
			return NVGRAPH_STATUS_ALLOC_FAILED; \
	}

#ifndef NVGRAPH_API
#ifdef _WIN32
#define NVGRAPH_API __stdcall
#else
#define NVGRAPH_API
#endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif

	/* nvGRAPH status type returns */
	typedef enum
	{
		NVGRAPH_STATUS_SUCCESS = 0,
		NVGRAPH_STATUS_NOT_INITIALIZED = 1,
		NVGRAPH_STATUS_ALLOC_FAILED = 2,
		NVGRAPH_STATUS_INVALID_VALUE = 3,
		NVGRAPH_STATUS_ARCH_MISMATCH = 4,
		NVGRAPH_STATUS_MAPPING_ERROR = 5,
		NVGRAPH_STATUS_EXECUTION_FAILED = 6,
		NVGRAPH_STATUS_INTERNAL_ERROR = 7,
		NVGRAPH_STATUS_TYPE_NOT_SUPPORTED = 8,
		NVGRAPH_STATUS_NOT_CONVERGED = 9,
		NVGRAPH_STATUS_GRAPH_TYPE_NOT_SUPPORTED = 10

	} nvgraphStatus_t;

	const char *nvgraphStatusGetString(nvgraphStatus_t status);

	/* Opaque structure holding nvGRAPH library context */
	struct nvgraphContext;
	typedef struct nvgraphContext *nvgraphHandle_t;

	/* Opaque structure holding the graph descriptor */
	struct nvgraphGraphDescr;
	typedef struct nvgraphGraphDescr *nvgraphGraphDescr_t;

	/* Semi-ring types */
	typedef enum
	{
		NVGRAPH_PLUS_TIMES_SR = 0,
		NVGRAPH_MIN_PLUS_SR = 1,
		NVGRAPH_MAX_MIN_SR = 2,
		NVGRAPH_OR_AND_SR = 3,
	} nvgraphSemiring_t;

	/* Topology types */
	typedef enum
	{
		NVGRAPH_CSR_32 = 0,
		NVGRAPH_CSC_32 = 1,
		NVGRAPH_COO_32 = 2,
		NVGRAPH_2D_32I_32I = 3,
		NVGRAPH_2D_64I_32I = 4
	} nvgraphTopologyType_t;

	typedef enum
	{
		NVGRAPH_DEFAULT = 0,			  // Default is unsorted.
		NVGRAPH_UNSORTED = 1,			  //
		NVGRAPH_SORTED_BY_SOURCE = 2,	  // CSR
		NVGRAPH_SORTED_BY_DESTINATION = 3 // CSC
	} nvgraphTag_t;

	typedef enum
	{
		NVGRAPH_MULTIPLY = 0,
		NVGRAPH_SUM = 1,
		NVGRAPH_MIN = 2,
		NVGRAPH_MAX = 3
	} nvgraphSemiringOps_t;

	typedef enum
	{
		NVGRAPH_MODULARITY_MAXIMIZATION = 0, // maximize modularity with Lanczos solver
		NVGRAPH_BALANCED_CUT_LANCZOS = 1,	 // minimize balanced cut with Lanczos solver
		NVGRAPH_BALANCED_CUT_LOBPCG = 2		 // minimize balanced cut with LOPCG solver
	} nvgraphSpectralClusteringType_t;

	struct SpectralClusteringParameter
	{
		int n_clusters;							   // number of clusters
		int n_eig_vects;						   // //number of eigenvectors
		nvgraphSpectralClusteringType_t algorithm; // algorithm to use
		float evs_tolerance;					   // tolerance of the eigensolver
		int evs_max_iter;						   // maximum number of iterations of the eigensolver
		float kmean_tolerance;					   // tolerance of kmeans
		int kmean_max_iter;						   // maximum number of iterations of kemeans
		void *opt;								   // optional parameter that can be used for preconditioning in the future
	};

	typedef enum
	{
		NVGRAPH_MODULARITY, // clustering score telling how good the clustering is compared to random assignment.
		NVGRAPH_EDGE_CUT,	// total number of edges between clusters.
		NVGRAPH_RATIO_CUT	// sum for all clusters of the number of edges going outside of the cluster divided by the number of vertex inside the cluster
	} nvgraphClusteringMetric_t;

	struct nvgraphCSRTopology32I_st
	{
		int nvertices;			  // n+1
		int nedges;				  // nnz
		int *source_offsets;	  // rowPtr
		int *destination_indices; // colInd
	};
	typedef struct nvgraphCSRTopology32I_st *nvgraphCSRTopology32I_t;

	struct nvgraphCSCTopology32I_st
	{
		int nvertices;			  // n+1
		int nedges;				  // nnz
		int *destination_offsets; // colPtr
		int *source_indices;	  // rowInd
	};
	typedef struct nvgraphCSCTopology32I_st *nvgraphCSCTopology32I_t;

	struct nvgraphCOOTopology32I_st
	{
		int nvertices;			  // n+1
		int nedges;				  // nnz
		int *source_indices;	  // rowInd
		int *destination_indices; // colInd
		nvgraphTag_t tag;
	};
	typedef struct nvgraphCOOTopology32I_st *nvgraphCOOTopology32I_t;

	struct nvgraph2dCOOTopology32I_st
	{
		int nvertices;
		int nedges;
		int *source_indices;		 // Row Indices
		int *destination_indices;	 // Column Indices
		hipblasDatatype_t valueType; // The type of values being given.
		void *values;				 // Pointer to array of values.
		int numDevices;				 // Gives the number of devices to be used.
		int *devices;				 // Array of device IDs to use.
		int blockN;					 // Specifies the value of n for an n x n matrix decomposition.
		nvgraphTag_t tag;
	};
	typedef struct nvgraph2dCOOTopology32I_st *nvgraph2dCOOTopology32I_t;

	/* Open the library and create the handle */
	nvgraphStatus_t NVGRAPH_API nvgraphCreate(nvgraphHandle_t *handle);

	/*  Close the library and destroy the handle  */
	nvgraphStatus_t NVGRAPH_API nvgraphDestroy(nvgraphHandle_t handle);

	/* Create an empty graph descriptor */
	nvgraphStatus_t NVGRAPH_API nvgraphCreateGraphDescr(nvgraphHandle_t handle,
														nvgraphGraphDescr_t *descrG);

	/* Destroy a graph descriptor */
	nvgraphStatus_t NVGRAPH_API nvgraphDestroyGraphDescr(nvgraphHandle_t handle,
														 nvgraphGraphDescr_t descrG);

	/* Set size, topology data in the graph descriptor  */
	nvgraphStatus_t NVGRAPH_API nvgraphSetGraphStructure(nvgraphHandle_t handle,
														 nvgraphGraphDescr_t descrG,
														 void *topologyData,
														 nvgraphTopologyType_t TType);

	/* Allocate numsets vectors of size V representing Vertex Data and attached them the graph.
	 * settypes[i] is the type of vector #i, currently all Vertex and Edge data should have the same type */
	nvgraphStatus_t NVGRAPH_API nvgraphAllocateVertexData(nvgraphHandle_t handle,
														  nvgraphGraphDescr_t descrG,
														  size_t numsets,
														  hipblasDatatype_t *settypes);

	/* Allocate numsets vectors of size E representing Edge Data and attached them the graph.
	 * settypes[i] is the type of vector #i, currently all Vertex and Edge data should have the same type */
	nvgraphStatus_t NVGRAPH_API nvgraphAllocateEdgeData(nvgraphHandle_t handle,
														nvgraphGraphDescr_t descrG,
														size_t numsets,
														hipblasDatatype_t *settypes);

	/* Update the vertex set #setnum with the data in *vertexData, sets have 0-based index
	 *  Conversions are not supported so nvgraphTopologyType_t should match the graph structure */
	nvgraphStatus_t NVGRAPH_API nvgraphSetVertexData(nvgraphHandle_t handle,
													 nvgraphGraphDescr_t descrG,
													 void *vertexData,
													 size_t setnum);

	/* Copy the edge set #setnum in *edgeData, sets have 0-based index
	 *  Conversions are not supported so nvgraphTopologyType_t should match the graph structure */
	nvgraphStatus_t NVGRAPH_API nvgraphGetVertexData(nvgraphHandle_t handle,
													 nvgraphGraphDescr_t descrG,
													 void *vertexData,
													 size_t setnum);

	/* Update the edge set #setnum with the data in *edgeData, sets have 0-based index
	 */
	nvgraphStatus_t NVGRAPH_API nvgraphSetEdgeData(nvgraphHandle_t handle,
												   nvgraphGraphDescr_t descrG,
												   void *edgeData,
												   size_t setnum);

	/* nvGRAPH PageRank
	 * Find PageRank for each vertex of a graph with a given transition probabilities, a bookmark vector of dangling vertices, and the damping factor.
	 */
	nvgraphStatus_t NVGRAPH_API nvgraphPagerank(nvgraphHandle_t handle,
												const nvgraphGraphDescr_t descrG,
												const size_t weight_index,
												const void *alpha,
												const size_t bookmark_index,
												const int has_guess,
												const size_t pagerank_index,
												const float tolerance,
												const int max_iter);

	/* nvGRAPH Triangles counting
	 * count number of triangles (cycles of size 3) formed by graph edges
	 */
	nvgraphStatus_t NVGRAPH_API nvgraphTriangleCount(nvgraphHandle_t handle,
													 const nvgraphGraphDescr_t graph_descr,
													 uint64_t *result);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _NVGRAPH_H_ */
