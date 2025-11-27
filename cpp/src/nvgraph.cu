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

#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cfloat>
#include <vector>
#include <nvlouvain.cuh>
#include <jaccard_gpu.cuh>
#include <hipblas.h>
#include <hipsolver.h>
#include <hip/hip_runtime.h>

#include <nvgraph_error.hxx>
#include <cnmem_shared_ptr.hxx>
#include <valued_csr_graph.hxx>
#include <multi_valued_csr_graph.hxx>
#include <nvgraph_vector.hxx>
#include <nvgraph_cusparse.hxx>
#include <nvgraph_cublas.hxx>
#include <nvgraph_csrmv.hxx>
#include <pagerank.hxx>
#include <arnoldi.hxx>
#include <sssp.hxx>
#include <widest_path.hxx>
#include <partition.hxx>
#include <nvgraph_convert.hxx>
#include <size2_selector.hxx>
#include <modularity_maximization.hxx>
#include <bfs.hxx>
#include <triangles_counting.hxx>

#include <csrmv_cub.h>

#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/system/hip/execution_policy.h>

#include <nvgraph.h>			  // public header **This is NVGRAPH C API**
#include <nvgraphP.h>			  // private header, contains structures, and potentially other things, used in the public C API that should never be exposed.
#include <nvgraph_experimental.h> // experimental header, contains hidden API entries, can be shared only under special circumstances without reveling internal things
#include "debug_macros.h"

#include "2d_partitioning.h"
#include "bfs2d.hxx"

static inline int check_context(const nvgraphHandle_t h)
{
	int ret = 0;
	if (h == NULL || !h->nvgraphIsInitialized)
		ret = 1;
	return ret;
}

static inline int check_graph(const nvgraphGraphDescr_t d)
{
	int ret = 0;
	if (d == NULL || d->graphStatus == IS_EMPTY)
		ret = 1;
	return ret;
}

static inline int check_int_size(size_t sz)
{
	int ret = 0;
	if (sz >= INT_MAX)
		ret = 1;
	return ret;
}

static inline int check_uniform_type_array(const hipblasDatatype_t *t, size_t sz)
{
	int ret = 0;
	hipblasDatatype_t uniform_type = t[0];
	for (size_t i = 1; i < sz; i++)
	{
		if (t[i] != uniform_type)
			ret = 1;
	}
	return ret;
}

template <typename T>
bool check_ptr(const T *p)
{
	bool ret = false;
	if (!p)
		ret = true;
	return ret;
}

#ifdef __cplusplus
extern "C"
{
#endif

	const char *nvgraphStatusGetString(nvgraphStatus_t status)
	{
		switch (status)
		{
		case NVGRAPH_STATUS_SUCCESS:
			return "Success";
		case NVGRAPH_STATUS_NOT_INITIALIZED:
			return "nvGRAPH not initialized";
		case NVGRAPH_STATUS_ALLOC_FAILED:
			return "nvGRAPH alloc failed";
		case NVGRAPH_STATUS_INVALID_VALUE:
			return "nvGRAPH invalid value";
		case NVGRAPH_STATUS_ARCH_MISMATCH:
			return "nvGRAPH arch mismatch";
		case NVGRAPH_STATUS_MAPPING_ERROR:
			return "nvGRAPH mapping error";
		case NVGRAPH_STATUS_EXECUTION_FAILED:
			return "nvGRAPH execution failed";
		case NVGRAPH_STATUS_INTERNAL_ERROR:
			return "nvGRAPH internal error";
		case NVGRAPH_STATUS_TYPE_NOT_SUPPORTED:
			return "nvGRAPH type not supported";
		case NVGRAPH_STATUS_NOT_CONVERGED:
			return "nvGRAPH algorithm failed to converge";
		case NVGRAPH_STATUS_GRAPH_TYPE_NOT_SUPPORTED:
			return "nvGRAPH graph type not supported";
		default:
			return "Unknown nvGRAPH Status";
		}
	};

#ifdef __cplusplus
} // extern "C"
#endif

namespace nvgraph
{

	// TODO: make those template functions in a separate header to be included by both
	// graph_extractor.cu and nvgraph.cpp;
	// right now this header does not exist and including graph_concrete_visitors.hxx
	// doesn't compile because of the Thrust code;
	
	nvgraphStatus_t getCAPIStatusForError(NVGRAPH_ERROR err)
	{
		nvgraphStatus_t ret = NVGRAPH_STATUS_SUCCESS;

		switch (err)
		{
		case NVGRAPH_OK:
			ret = NVGRAPH_STATUS_SUCCESS;
			break;
		case NVGRAPH_ERR_BAD_PARAMETERS:
			ret = NVGRAPH_STATUS_INVALID_VALUE;
			break;
		case NVGRAPH_ERR_UNKNOWN:
			ret = NVGRAPH_STATUS_INTERNAL_ERROR;
			break;
		case NVGRAPH_ERR_CUDA_FAILURE:
			ret = NVGRAPH_STATUS_EXECUTION_FAILED;
			break;
		case NVGRAPH_ERR_THRUST_FAILURE:
			ret = NVGRAPH_STATUS_EXECUTION_FAILED;
			break;
		case NVGRAPH_ERR_IO:
			ret = NVGRAPH_STATUS_INTERNAL_ERROR;
			break;
		case NVGRAPH_ERR_NOT_IMPLEMENTED:
			ret = NVGRAPH_STATUS_INVALID_VALUE;
			break;
		case NVGRAPH_ERR_NO_MEMORY:
			ret = NVGRAPH_STATUS_ALLOC_FAILED;
			break;
		case NVGRAPH_ERR_NOT_CONVERGED:
			ret = NVGRAPH_STATUS_NOT_CONVERGED;
			break;
		default:
			ret = NVGRAPH_STATUS_INTERNAL_ERROR;
		}
		return ret;
	}

	static nvgraphStatus_t nvgraphCreateMulti_impl(struct nvgraphContext **outCtx,
												   int numDevices,
												   int *_devices)
	{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			int device;

			CHECK_CUDA(hipFree((void *)0));
			CHECK_CUDA(hipGetDevice(&device));
			struct nvgraphContext *ctx = NULL;
			ctx = (struct nvgraphContext *)malloc(sizeof(*ctx));
			if (!ctx)
			{
				FatalError("Cannot allocate NVGRAPH context.", NVGRAPH_ERR_UNKNOWN);
			}

			// cnmem
			memset(&ctx->cnmem_device, 0, sizeof(ctx->cnmem_device)); // init all to 0
			ctx->cnmem_device.device = device;						  // cnmem runs on the device set by hipSetDevice

			size_t init_alloc = 1; // Initial allocation tentative, it is currently 1 so this feature is basically disabeled.

			// Warning : Should uncomment that if using init_alloc > 1
			// size_t freeMem, totalMem;
			// hipMemGetInfo(&freeMem, &totalMem);
			// if (freeMem < init_alloc) // Couldn't find enough memory to do the initial alloc
			//    init_alloc = 1; // (0 is used as default parameter in cnmem)

			ctx->cnmem_device.size = init_alloc;
			cnmemDevice_t *devices = (cnmemDevice_t *)malloc(sizeof(cnmemDevice_t) * numDevices);
			memset(devices, 0, sizeof(cnmemDevice_t) * numDevices);
			for (int i = 0; i < numDevices; i++)
			{
				devices[i].device = _devices[i];
				devices[i].size = 1;
			}
			cnmemStatus_t cm_status = cnmemInit(numDevices, devices, CNMEM_FLAGS_DEFAULT);
			free(devices);
			if (cm_status != CNMEM_STATUS_SUCCESS)
				FatalError("Cannot initialize memory manager.", NVGRAPH_ERR_UNKNOWN);

			// Cublas and Cusparse
			Cusparse::get_handle();
			Cublas::get_handle();

			// others
			ctx->stream = 0;
			ctx->nvgraphIsInitialized = true;

			if (outCtx)
			{
				*outCtx = ctx;
			}
		}
		NVGRAPH_CATCHES(rc)

		return getCAPIStatusForError(rc);
	}

	static nvgraphStatus_t nvgraphCreate_impl(struct nvgraphContext **outCtx)
	{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			int device;

			CHECK_CUDA(hipFree((void *)0));
			CHECK_CUDA(hipGetDevice(&device));
			struct nvgraphContext *ctx = NULL;
			ctx = (struct nvgraphContext *)malloc(sizeof(*ctx));
			if (!ctx)
			{
				FatalError("Cannot allocate NVGRAPH context.", NVGRAPH_ERR_UNKNOWN);
			}

			// cnmem
			memset(&ctx->cnmem_device, 0, sizeof(ctx->cnmem_device)); // init all to 0
			ctx->cnmem_device.device = device;						  // cnmem runs on the device set by hipSetDevice

			size_t init_alloc = 1; // Initial allocation tentative, it is currently 1 so this feature is basically disabeled.

			// Warning : Should uncomment that if using init_alloc > 1
			// size_t freeMem, totalMem;
			// hipMemGetInfo(&freeMem, &totalMem);
			// if (freeMem < init_alloc) // Couldn't find enough memory to do the initial alloc
			//    init_alloc = 1; // (0 is used as default parameter in cnmem)

			ctx->cnmem_device.size = init_alloc;

			cnmemStatus_t cm_status = cnmemInit(1, &ctx->cnmem_device, CNMEM_FLAGS_DEFAULT);
			if (cm_status != CNMEM_STATUS_SUCCESS)
				FatalError("Cannot initialize memory manager.", NVGRAPH_ERR_UNKNOWN);

			// Cublas and Cusparse
			Cusparse::get_handle();
			Cublas::get_handle();

			// others
			ctx->stream = 0;
			ctx->nvgraphIsInitialized = true;

			if (outCtx)
			{
				*outCtx = ctx;
			}
		}
		NVGRAPH_CATCHES(rc)

		return getCAPIStatusForError(rc);
	}

	static nvgraphStatus_t nvgraphDestroy_impl(nvgraphHandle_t handle)
	{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle))
				FatalError("Cannot initialize memory manager.", NVGRAPH_ERR_NO_MEMORY);

			// Cublas and Cusparse
			Cusparse::destroy_handle();
			Cublas::destroy_handle();
			// cnmem

//     compiler is complaining, cm_status is not used in release build
#ifdef DEBUG
			cnmemStatus_t cm_status = cnmemFinalize();
			if (cm_status != CNMEM_STATUS_SUCCESS)
			{
				CERR() << "Warning: " << cnmemGetErrorString(cm_status) << std::endl;
			}
#else
			cnmemFinalize();
#endif
			// others
			free(handle);
		}
		NVGRAPH_CATCHES(rc)

		return getCAPIStatusForError(rc);
	}

	static nvgraphStatus_t nvgraphCreateGraphDescr_impl(nvgraphHandle_t handle,
														struct nvgraphGraphDescr **outGraphDescr)
	{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

			struct nvgraphGraphDescr *descrG = NULL;
			descrG = (struct nvgraphGraphDescr *)malloc(sizeof(*descrG));
			if (!descrG)
			{
				FatalError("Cannot allocate graph descriptor.", NVGRAPH_ERR_UNKNOWN);
			}
			descrG->graphStatus = IS_EMPTY;
			if (outGraphDescr)
			{
				*outGraphDescr = descrG;
			}
		}
		NVGRAPH_CATCHES(rc)

		return getCAPIStatusForError(rc);
	}

	static nvgraphStatus_t nvgraphDestroyGraphDescr_impl(nvgraphHandle_t handle,
														 struct nvgraphGraphDescr *descrG)
	{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (descrG)
			{
				if (descrG->TT == NVGRAPH_2D_32I_32I)
				{
					switch (descrG->T)
					{
					case HIPBLAS_R_32I:
					{
						Matrix2d<int32_t, int32_t, int32_t> *m =
							static_cast<Matrix2d<int32_t, int32_t, int32_t> *>(descrG->graph_handle);
						delete m;
						break;
					}
					default:
						return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
					}
				}
				else
				{
					switch (descrG->graphStatus)
					{
					case IS_EMPTY:
					{
						break;
					}
					case HAS_TOPOLOGY:
					{
						CsrGraph<int> *CSRG =
							static_cast<CsrGraph<int> *>(descrG->graph_handle);
						delete CSRG;
						break;
					}
					case HAS_VALUES:
					{
						if (descrG->T == HIPBLAS_R_32F)
						{
							MultiValuedCsrGraph<int, float> *MCSRG =
								static_cast<MultiValuedCsrGraph<int, float> *>(descrG->graph_handle);
							delete MCSRG;
						}
						else if (descrG->T == HIPBLAS_R_64F)
						{
							MultiValuedCsrGraph<int, double> *MCSRG =
								static_cast<MultiValuedCsrGraph<int, double> *>(descrG->graph_handle);
							delete MCSRG;
						}
						else if (descrG->T == HIPBLAS_R_32I)
						{
							MultiValuedCsrGraph<int, int> *MCSRG =
								static_cast<MultiValuedCsrGraph<int, int> *>(descrG->graph_handle);
							delete MCSRG;
						}
						else
							return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
						break;
					}
					default:
						return NVGRAPH_STATUS_INVALID_VALUE;
					}
				}
				free(descrG);
			}
			else
				return NVGRAPH_STATUS_INVALID_VALUE;
		}
		NVGRAPH_CATCHES(rc)

		return getCAPIStatusForError(rc);
	}

	nvgraphStatus_t NVGRAPH_API nvgraphSetGraphStructure_impl(nvgraphHandle_t handle,
															  nvgraphGraphDescr_t descrG,
															  void *topologyData,
															  nvgraphTopologyType_t TT)
	{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
			if (descrG->graphStatus != IS_EMPTY)
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
			if (check_ptr(topologyData))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (TT == NVGRAPH_CSR_32 || TT == NVGRAPH_CSC_32)
			{
				int v = 0, e = 0, *neighborhood = NULL, *edgedest = NULL;
				switch (TT)
				{
				case NVGRAPH_CSR_32:
				{
					nvgraphCSRTopology32I_t t = static_cast<nvgraphCSRTopology32I_t>(topologyData);
					if (!t->nvertices || !t->nedges || check_ptr(t->source_offsets) || check_ptr(t->destination_indices))
						return NVGRAPH_STATUS_INVALID_VALUE;
					v = t->nvertices;
					e = t->nedges;
					neighborhood = t->source_offsets;
					edgedest = t->destination_indices;
					break;
				}
				case NVGRAPH_CSC_32:
				{
					nvgraphCSCTopology32I_t t = static_cast<nvgraphCSCTopology32I_t>(topologyData);
					if (!t->nvertices || !t->nedges || check_ptr(t->destination_offsets) || check_ptr(t->source_indices))
						return NVGRAPH_STATUS_INVALID_VALUE;
					v = t->nvertices;
					e = t->nedges;
					neighborhood = t->destination_offsets;
					edgedest = t->source_indices;
					break;
				}
				default:
					return NVGRAPH_STATUS_INVALID_VALUE;
				}

				descrG->TT = TT;

				// Create the internal CSR representation
				CsrGraph<int> *CSRG = new CsrGraph<int>(v, e, handle->stream);

				CHECK_CUDA(hipMemcpy(CSRG->get_raw_row_offsets(),
									 neighborhood,
									 (size_t)((CSRG->get_num_vertices() + 1) * sizeof(int)),
									 hipMemcpyDefault));

				CHECK_CUDA(hipMemcpy(CSRG->get_raw_column_indices(),
									 edgedest,
									 (size_t)((CSRG->get_num_edges()) * sizeof(int)),
									 hipMemcpyDefault));

				// Set the graph handle
				descrG->graph_handle = CSRG;
				descrG->graphStatus = HAS_TOPOLOGY;
			}
			else if (TT == NVGRAPH_2D_32I_32I)
			{
				nvgraph2dCOOTopology32I_t td = static_cast<nvgraph2dCOOTopology32I_t>(topologyData);
				switch (td->valueType)
				{
				case HIPBLAS_R_32I:
				{
					if (!td->nvertices || !td->nedges || !td->source_indices || !td->destination_indices || !td->numDevices || !td->devices || !td->blockN)
						return NVGRAPH_STATUS_INVALID_VALUE;
					descrG->TT = TT;
					descrG->graphStatus = HAS_TOPOLOGY;
					if (td->values)
						descrG->graphStatus = HAS_VALUES;
					descrG->T = td->valueType;
					std::vector<int32_t> devices;
					for (int32_t i = 0; i < td->numDevices; i++)
						devices.push_back(td->devices[i]);
					MatrixDecompositionDescription<int32_t, int32_t> description(td->nvertices,
																				 td->blockN,
																				 td->nedges,
																				 devices);
					Matrix2d<int32_t, int32_t, int32_t> *m = new Matrix2d<int32_t,
																		  int32_t, int32_t>();
					*m = COOto2d(description,
								 td->source_indices,
								 td->destination_indices,
								 (int32_t *)td->values);
					descrG->graph_handle = m;
					break;
				}
				default:
				{
					return NVGRAPH_STATUS_INVALID_VALUE;
				}
				}
			}
			else
			{
				return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
			}
		}
		NVGRAPH_CATCHES(rc)

		return getCAPIStatusForError(rc);
	}

	nvgraphStatus_t NVGRAPH_API nvgraphAllocateVertexData_impl(nvgraphHandle_t handle,
															   nvgraphGraphDescr_t descrG,
															   size_t numsets,
															   hipblasDatatype_t *settypes)
	{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle) || check_graph(descrG) || check_int_size(numsets) || check_ptr(settypes))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
			if (check_uniform_type_array(settypes, numsets))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (descrG->graphStatus == HAS_TOPOLOGY) // need to convert CsrGraph to MultiValuedCsrGraph first
			{
				if (*settypes == HIPBLAS_R_32F)
				{
					CsrGraph<int> *CSRG =
						static_cast<CsrGraph<int> *>(descrG->graph_handle);
					MultiValuedCsrGraph<int, float> *MCSRG = new MultiValuedCsrGraph<
						int, float>(*CSRG);
					descrG->graph_handle = MCSRG;
				}
				else if (*settypes == HIPBLAS_R_64F)
				{
					CsrGraph<int> *CSRG =
						static_cast<CsrGraph<int> *>(descrG->graph_handle);
					MultiValuedCsrGraph<int, double> *MCSRG = new MultiValuedCsrGraph<
						int, double>(*CSRG);
					descrG->graph_handle = MCSRG;
				}
				else if (*settypes == HIPBLAS_R_32I)
				{
					CsrGraph<int> *CSRG =
						static_cast<CsrGraph<int> *>(descrG->graph_handle);
					MultiValuedCsrGraph<int, int> *MCSRG = new MultiValuedCsrGraph<int,
																				   int>(*CSRG);
					descrG->graph_handle = MCSRG;
				}
				else
					return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
				descrG->T = *settypes;
				descrG->graphStatus = HAS_VALUES;
			}
			else if (descrG->graphStatus == HAS_VALUES) // Already in MultiValuedCsrGraph, just need to check the type
			{
				if (*settypes != descrG->T)
					return NVGRAPH_STATUS_INVALID_VALUE;
			}
			else
				return NVGRAPH_STATUS_INVALID_VALUE;

			// Allocate and transfer
			if (*settypes == HIPBLAS_R_32F)
			{
				MultiValuedCsrGraph<int, float> *MCSRG =
					static_cast<MultiValuedCsrGraph<int, float> *>(descrG->graph_handle);
				MCSRG->allocateVertexData(numsets, NULL);
			}
			else if (*settypes == HIPBLAS_R_64F)
			{
				MultiValuedCsrGraph<int, double> *MCSRG =
					static_cast<MultiValuedCsrGraph<int, double> *>(descrG->graph_handle);
				MCSRG->allocateVertexData(numsets, NULL);
			}
			else if (*settypes == HIPBLAS_R_32I)
			{
				MultiValuedCsrGraph<int, int> *MCSRG =
					static_cast<MultiValuedCsrGraph<int, int> *>(descrG->graph_handle);
				MCSRG->allocateVertexData(numsets, NULL);
			}
			else
				return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
		}
		NVGRAPH_CATCHES(rc)

		return getCAPIStatusForError(rc);
	}

	nvgraphStatus_t NVGRAPH_API nvgraphAllocateEdgeData_impl(nvgraphHandle_t handle,
															 nvgraphGraphDescr_t descrG,
															 size_t numsets,
															 hipblasDatatype_t *settypes)
	{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle) || check_graph(descrG) || check_int_size(numsets) || check_ptr(settypes))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
			if (check_uniform_type_array(settypes, numsets))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
			// Look at what kind of graph we have
			if (descrG->graphStatus == HAS_TOPOLOGY) // need to convert CsrGraph to MultiValuedCsrGraph first
			{
				if (*settypes == HIPBLAS_R_32F)
				{
					CsrGraph<int> *CSRG =
						static_cast<CsrGraph<int> *>(descrG->graph_handle);
					MultiValuedCsrGraph<int, float> *MCSRG = new MultiValuedCsrGraph<
						int, float>(*CSRG);
					descrG->graph_handle = MCSRG;
				}
				else if (*settypes == HIPBLAS_R_64F)
				{
					CsrGraph<int> *CSRG =
						static_cast<CsrGraph<int> *>(descrG->graph_handle);
					MultiValuedCsrGraph<int, double> *MCSRG = new MultiValuedCsrGraph<
						int, double>(*CSRG);
					descrG->graph_handle = MCSRG;
				}
				else if (*settypes == HIPBLAS_R_32I)
				{
					CsrGraph<int> *CSRG =
						static_cast<CsrGraph<int> *>(descrG->graph_handle);
					MultiValuedCsrGraph<int, int> *MCSRG = new MultiValuedCsrGraph<int,
																				   int>(*CSRG);
					descrG->graph_handle = MCSRG;
				}
				else
					return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
				descrG->T = *settypes;
				descrG->graphStatus = HAS_VALUES;
			}
			else if (descrG->graphStatus == HAS_VALUES) // Already in MultiValuedCsrGraph, just need to check the type
			{
				if (*settypes != descrG->T)
					return NVGRAPH_STATUS_INVALID_VALUE;
			}
			else
				return NVGRAPH_STATUS_INVALID_VALUE;

			// Allocate and transfer
			if (*settypes == HIPBLAS_R_32F)
			{
				MultiValuedCsrGraph<int, float> *MCSRG =
					static_cast<MultiValuedCsrGraph<int, float> *>(descrG->graph_handle);
				MCSRG->allocateEdgeData(numsets, NULL);
			}
			else if (*settypes == HIPBLAS_R_64F)
			{
				MultiValuedCsrGraph<int, double> *MCSRG =
					static_cast<MultiValuedCsrGraph<int, double> *>(descrG->graph_handle);
				MCSRG->allocateEdgeData(numsets, NULL);
			}
			else if (*settypes == HIPBLAS_R_32I)
			{
				MultiValuedCsrGraph<int, int> *MCSRG =
					static_cast<MultiValuedCsrGraph<int, int> *>(descrG->graph_handle);
				MCSRG->allocateEdgeData(numsets, NULL);
			}
			else
				return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
		}
		NVGRAPH_CATCHES(rc)

		return getCAPIStatusForError(rc);
	}

	nvgraphStatus_t NVGRAPH_API nvgraphSetVertexData_impl(nvgraphHandle_t handle,
														  nvgraphGraphDescr_t descrG,
														  void *vertexData,
														  size_t setnum)
	{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle) || check_graph(descrG) || check_int_size(setnum) || check_ptr(vertexData))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
				FatalError("Graph should have allocated values.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (descrG->T == HIPBLAS_R_32F)
			{
				MultiValuedCsrGraph<int, float> *MCSRG =
					static_cast<MultiValuedCsrGraph<int, float> *>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				hipMemcpy(MCSRG->get_raw_vertex_dim(setnum),
						  (float *)vertexData,
						  (size_t)((MCSRG->get_num_vertices()) * sizeof(float)),
						  hipMemcpyDefault);
			}
			else if (descrG->T == HIPBLAS_R_64F)
			{
				MultiValuedCsrGraph<int, double> *MCSRG =
					static_cast<MultiValuedCsrGraph<int, double> *>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				hipMemcpy(MCSRG->get_raw_vertex_dim(setnum),
						  (double *)vertexData,
						  (size_t)((MCSRG->get_num_vertices()) * sizeof(double)),
						  hipMemcpyDefault);
			}
			else if (descrG->T == HIPBLAS_R_32I)
			{
				MultiValuedCsrGraph<int, int> *MCSRG =
					static_cast<MultiValuedCsrGraph<int, int> *>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				hipMemcpy(MCSRG->get_raw_vertex_dim(setnum),
						  (int *)vertexData,
						  (size_t)((MCSRG->get_num_vertices()) * sizeof(int)),
						  hipMemcpyDefault);
			}
			else
				return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;

			cudaCheckError();
		}
		NVGRAPH_CATCHES(rc)

		return getCAPIStatusForError(rc);
	}

	nvgraphStatus_t NVGRAPH_API nvgraphGetVertexData_impl(nvgraphHandle_t handle,
														  nvgraphGraphDescr_t descrG,
														  void *vertexData,
														  size_t setnum)
	{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle) || check_graph(descrG) || check_int_size(setnum) || check_ptr(vertexData))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
				FatalError("Graph should have values.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (descrG->T == HIPBLAS_R_32F)
			{
				MultiValuedCsrGraph<int, float> *MCSRG =
					static_cast<MultiValuedCsrGraph<int, float> *>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				hipMemcpy((float *)vertexData,
						  MCSRG->get_raw_vertex_dim(setnum),
						  (size_t)((MCSRG->get_num_vertices()) * sizeof(float)),
						  hipMemcpyDefault);
			}
			else if (descrG->T == HIPBLAS_R_64F)
			{
				MultiValuedCsrGraph<int, double> *MCSRG =
					static_cast<MultiValuedCsrGraph<int, double> *>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				hipMemcpy((double *)vertexData,
						  MCSRG->get_raw_vertex_dim(setnum),
						  (size_t)((MCSRG->get_num_vertices()) * sizeof(double)),
						  hipMemcpyDefault);
			}
			else if (descrG->T == HIPBLAS_R_32I)
			{
				MultiValuedCsrGraph<int, int> *MCSRG =
					static_cast<MultiValuedCsrGraph<int, int> *>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				hipMemcpy((int *)vertexData,
						  MCSRG->get_raw_vertex_dim(setnum),
						  (size_t)((MCSRG->get_num_vertices()) * sizeof(int)),
						  hipMemcpyDefault);
			}
			else
				return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;

			cudaCheckError();
		}
		NVGRAPH_CATCHES(rc)

		return getCAPIStatusForError(rc);
	}

	nvgraphStatus_t NVGRAPH_API nvgraphSetEdgeData_impl(nvgraphHandle_t handle,
														nvgraphGraphDescr_t descrG,
														void *edgeData,
														size_t setnum)
	{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle) || check_graph(descrG) || check_int_size(setnum) || check_ptr(edgeData))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
				return NVGRAPH_STATUS_INVALID_VALUE;

			if (descrG->T == HIPBLAS_R_32F)
			{
				MultiValuedCsrGraph<int, float> *MCSRG =
					static_cast<MultiValuedCsrGraph<int, float> *>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_edge_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				hipMemcpy(MCSRG->get_raw_edge_dim(setnum),
						  (float *)edgeData,
						  (size_t)((MCSRG->get_num_edges()) * sizeof(float)),
						  hipMemcpyDefault);
			}
			else if (descrG->T == HIPBLAS_R_64F)
			{
				MultiValuedCsrGraph<int, double> *MCSRG =
					static_cast<MultiValuedCsrGraph<int, double> *>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_edge_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				hipMemcpy(MCSRG->get_raw_edge_dim(setnum),
						  (double *)edgeData,
						  (size_t)((MCSRG->get_num_edges()) * sizeof(double)),
						  hipMemcpyDefault);
			}
			else if (descrG->T == HIPBLAS_R_32I)
			{
				MultiValuedCsrGraph<int, int> *MCSRG =
					static_cast<MultiValuedCsrGraph<int, int> *>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_edge_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				hipMemcpy(MCSRG->get_raw_edge_dim(setnum),
						  (int *)edgeData,
						  (size_t)((MCSRG->get_num_edges()) * sizeof(int)),
						  hipMemcpyDefault);
			}
			else
				return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;

			cudaCheckError();
		}
		NVGRAPH_CATCHES(rc)

		return getCAPIStatusForError(rc);
	}

	nvgraphStatus_t NVGRAPH_API nvgraphPagerank_impl(nvgraphHandle_t handle,
													 const nvgraphGraphDescr_t descrG,
													 const size_t weight_index,
													 const void *alpha,
													 const size_t bookmark,
													 const int has_guess,
													 const size_t rank,
													 const float tolerance,
													 const int max_iter)
	{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle) || check_graph(descrG) || check_int_size(weight_index) || check_ptr(alpha))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
				return NVGRAPH_STATUS_INVALID_VALUE;

			if (descrG->TT != NVGRAPH_CSC_32) // supported topologies
				return NVGRAPH_STATUS_INVALID_VALUE;

			if (!(has_guess == 0 || has_guess == 1))
				return NVGRAPH_STATUS_INVALID_VALUE;

			int max_it;
			float tol;

			if (max_iter > 0)
				max_it = max_iter;
			else
				max_it = 500;

			if (tolerance == 0.0f)
				tol = 1.0E-6f;
			else if (tolerance < 1.0f && tolerance > 0.0f)
				tol = tolerance;
			else
				return NVGRAPH_STATUS_INVALID_VALUE;

			switch (descrG->T)
			{
			case HIPBLAS_R_32F:
			{
				float alphaT = *static_cast<const float *>(alpha);
				if (alphaT <= 0.0f || alphaT >= 1.0f)
					return NVGRAPH_STATUS_INVALID_VALUE;
				MultiValuedCsrGraph<int, float> *MCSRG =
					static_cast<MultiValuedCsrGraph<int, float> *>(descrG->graph_handle);
				if (weight_index >= MCSRG->get_num_edge_dim() || bookmark >= MCSRG->get_num_vertex_dim() || rank >= MCSRG->get_num_vertex_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;

				int n = static_cast<int>(MCSRG->get_num_vertices());
				Vector<float> guess(n, handle->stream);
				Vector<float> bm(n, handle->stream);
				if (has_guess)
					guess.copy(MCSRG->get_vertex_dim(rank));
				else
					guess.fill(static_cast<float>(1.0 / n));
				bm.copy(MCSRG->get_vertex_dim(bookmark));
				Pagerank<int, float> pagerank_solver(*MCSRG->get_valued_csr_graph(weight_index),
													 bm);
				rc = pagerank_solver.solve(alphaT, guess, MCSRG->get_vertex_dim(rank), tol, max_it);
				break;
			}
			case HIPBLAS_R_64F:
			{
				double alphaT = *static_cast<const double *>(alpha);
				if (alphaT <= 0.0 || alphaT >= 1.0)
					return NVGRAPH_STATUS_INVALID_VALUE;

				MultiValuedCsrGraph<int, double> *MCSRG =
					static_cast<MultiValuedCsrGraph<int, double> *>(descrG->graph_handle);
				if (weight_index >= MCSRG->get_num_edge_dim() || bookmark >= MCSRG->get_num_vertex_dim() || rank >= MCSRG->get_num_vertex_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;

				int n = static_cast<int>(MCSRG->get_num_vertices());
				Vector<double> guess(n, handle->stream);
				Vector<double> bm(n, handle->stream);
				bm.copy(MCSRG->get_vertex_dim(bookmark));
				if (has_guess)
					guess.copy(MCSRG->get_vertex_dim(rank));
				else
					guess.fill(static_cast<float>(1.0 / n));
				Pagerank<int, double> pagerank_solver(*MCSRG->get_valued_csr_graph(weight_index),
													  bm);
				rc = pagerank_solver.solve(alphaT, guess, MCSRG->get_vertex_dim(rank), tol, max_it);
				break;
			}
			default:
				return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
			}
		}
		NVGRAPH_CATCHES(rc)

		return getCAPIStatusForError(rc);
	}

	nvgraphStatus_t NVGRAPH_API nvgraphTriangleCount_impl(nvgraphHandle_t handle,
														  const nvgraphGraphDescr_t descrG,
														  uint64_t *result)
	{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle) || check_graph(descrG) || check_ptr(result))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (descrG->TT != NVGRAPH_CSR_32 && descrG->TT != NVGRAPH_CSC_32) // supported topologies
				return NVGRAPH_STATUS_INVALID_VALUE;

			if (descrG->graphStatus != HAS_TOPOLOGY && descrG->graphStatus != HAS_VALUES)
			{
				return NVGRAPH_STATUS_INVALID_VALUE; // should have topology
			}

			CsrGraph<int> *CSRG = static_cast<CsrGraph<int> *>(descrG->graph_handle);
			if (CSRG == NULL)
				return NVGRAPH_STATUS_MAPPING_ERROR;
			triangles_counting::TrianglesCount<int> counter(*CSRG); /* stream, device */
			rc = counter.count();
			uint64_t s_res = counter.get_triangles_count();
			*result = static_cast<uint64_t>(s_res);
		}
		NVGRAPH_CATCHES(rc)
		return getCAPIStatusForError(rc);
	}

} /*namespace nvgraph*/

/*************************
 *        API
 *************************/

nvgraphStatus_t NVGRAPH_API nvgraphCreate(nvgraphHandle_t *handle)
{
	return nvgraph::nvgraphCreate_impl(handle);
}

nvgraphStatus_t NVGRAPH_API nvgraphDestroy(nvgraphHandle_t handle)
{
	return nvgraph::nvgraphDestroy_impl(handle);
}

nvgraphStatus_t NVGRAPH_API nvgraphCreateGraphDescr(nvgraphHandle_t handle,
													nvgraphGraphDescr_t *descrG)
{
	return nvgraph::nvgraphCreateGraphDescr_impl(handle, descrG);
}

nvgraphStatus_t NVGRAPH_API nvgraphDestroyGraphDescr(nvgraphHandle_t handle,
													 nvgraphGraphDescr_t descrG)
{
	return nvgraph::nvgraphDestroyGraphDescr_impl(handle, descrG);
}

nvgraphStatus_t NVGRAPH_API nvgraphAllocateVertexData(nvgraphHandle_t handle,
													  nvgraphGraphDescr_t descrG,
													  size_t numsets,
													  hipblasDatatype_t *settypes)
{
	return nvgraph::nvgraphAllocateVertexData_impl(handle, descrG, numsets, settypes);
}

nvgraphStatus_t NVGRAPH_API nvgraphAllocateEdgeData(nvgraphHandle_t handle,
													nvgraphGraphDescr_t descrG,
													size_t numsets,
													hipblasDatatype_t *settypes)
{
	return nvgraph::nvgraphAllocateEdgeData_impl(handle, descrG, numsets, settypes);
}

nvgraphStatus_t NVGRAPH_API nvgraphSetVertexData(nvgraphHandle_t handle,
												 nvgraphGraphDescr_t descrG,
												 void *vertexData,
												 size_t setnum)
{
	return nvgraph::nvgraphSetVertexData_impl(handle, descrG, vertexData, setnum);
}

nvgraphStatus_t NVGRAPH_API nvgraphGetVertexData(nvgraphHandle_t handle,
												 nvgraphGraphDescr_t descrG,
												 void *vertexData,
												 size_t setnum)
{
	return nvgraph::nvgraphGetVertexData_impl(handle, descrG, vertexData, setnum);
}

nvgraphStatus_t NVGRAPH_API nvgraphSetEdgeData(nvgraphHandle_t handle,
											   nvgraphGraphDescr_t descrG,
											   void *edgeData,
											   size_t setnum)
{
	return nvgraph::nvgraphSetEdgeData_impl(handle, descrG, edgeData, setnum);
}

nvgraphStatus_t NVGRAPH_API nvgraphPagerank(nvgraphHandle_t handle,
											const nvgraphGraphDescr_t descrG,
											const size_t weight_index,
											const void *alpha,
											const size_t bookmark,
											const int has_guess,
											const size_t pagerank_index,
											const float tolerance,
											const int max_iter)
{
	return nvgraph::nvgraphPagerank_impl(handle,
										 descrG,
										 weight_index,
										 alpha,
										 bookmark,
										 has_guess,
										 pagerank_index,
										 tolerance,
										 max_iter);
}

nvgraphStatus_t NVGRAPH_API nvgraphTriangleCount(nvgraphHandle_t handle,
												 const nvgraphGraphDescr_t descrG,
												 uint64_t *result)
{
	return nvgraph::nvgraphTriangleCount_impl(handle, descrG, result);
}

nvgraphStatus_t NVGRAPH_API nvgraphAttachVertexData(nvgraphHandle_t handle,
													nvgraphGraphDescr_t descrG,
													size_t setnum,
													hipblasDatatype_t settype,
													void *vertexData)
{
	return nvgraph::nvgraphAttachVertexData_impl(handle, descrG, setnum, settype, vertexData);
}
