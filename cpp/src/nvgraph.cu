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
#include <cusolverDn.h>

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
#include <nvgraph_convert.hxx>
#include <size2_selector.hxx>
#include <triangles_counting.hxx>

#include <csrmv_cub.h>

#include <nvgraph.h>   // public header **This is NVGRAPH C API**
#include <nvgraphP.h>  // private header, contains structures, and potentially other things, used in the public C API that should never be exposed.
#include <nvgraph_experimental.h>  // experimental header, contains hidden API entries, can be shared only under special circumstances without reveling internal things
#include "debug_macros.h"

#include "2d_partitioning.h"

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
static inline int check_topology(const nvgraphGraphDescr_t d)
											{
	int ret = 0;
	if (d->graphStatus == IS_EMPTY)
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

static inline int check_int_ptr(const int* p)
											{
	int ret = 0;
	if (!p)
		ret = 1;
	return ret;
}

static inline int check_uniform_type_array(const cudaDataType_t * t, size_t sz)
															{
	int ret = 0;
	cudaDataType_t uniform_type = t[0];
	for (size_t i = 1; i < sz; i++)
			{
		if (t[i] != uniform_type)
			ret = 1;
	}
	return ret;
}

template<typename T>
bool check_ptr(const T* p)
					{
	bool ret = false;
	if (!p)
		ret = true;
	return ret;
}

namespace nvgraph
{

//TODO: make those template functions in a separate header to be included by both
//graph_extractor.cu and nvgraph.cpp;
//right now this header does not exist and including graph_concrete_visitors.hxx
//doesn't compile because of the Thrust code;
//
	extern CsrGraph<int>* extract_subgraph_by_vertices(CsrGraph<int>& graph,
																		int* pV,
																		size_t n,
																		cudaStream_t stream);
	extern MultiValuedCsrGraph<int, float>* extract_subgraph_by_vertices(MultiValuedCsrGraph<int,
																										float>& graph,
																								int* pV,
																								size_t n,
																								cudaStream_t stream);
	extern MultiValuedCsrGraph<int, double>* extract_subgraph_by_vertices(MultiValuedCsrGraph<int,
																											double>& graph,
																									int* pV,
																									size_t n,
																									cudaStream_t stream);

	extern CsrGraph<int>* extract_subgraph_by_edges(CsrGraph<int>& graph,
																	int* pV,
																	size_t n,
																	cudaStream_t stream);
	extern MultiValuedCsrGraph<int, float>* extract_subgraph_by_edges(MultiValuedCsrGraph<int, float>& graph,
																							int* pV,
																							size_t n,
																							cudaStream_t stream);
	extern MultiValuedCsrGraph<int, double>* extract_subgraph_by_edges(MultiValuedCsrGraph<int,
																										double>& graph,
																								int* pV,
																								size_t n,
																								cudaStream_t stream);

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

	extern "C" {
		const char* nvgraphStatusGetString(nvgraphStatus_t status)
														{
			switch (status) {
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
		}
		;
	}

	static nvgraphStatus_t nvgraphCreateMulti_impl(struct nvgraphContext **outCtx,
																	int numDevices,
																	int* _devices) {
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			int device;

			CHECK_CUDA(cudaFree((void * )0));
			CHECK_CUDA(cudaGetDevice(&device));
			struct nvgraphContext *ctx = NULL;
			ctx = (struct nvgraphContext *) malloc(sizeof(*ctx));
			if (!ctx) {
				FatalError("Cannot allocate NVGRAPH context.", NVGRAPH_ERR_UNKNOWN);
			}

			//cnmem
			memset(&ctx->cnmem_device, 0, sizeof(ctx->cnmem_device)); // init all to 0
			ctx->cnmem_device.device = device; // cnmem runs on the device set by cudaSetDevice

			size_t init_alloc = 1; // Initial allocation tentative, it is currently 1 so this feature is basically disabeled.

			// Warning : Should uncomment that if using init_alloc > 1
			//size_t freeMem, totalMem;
			//cudaMemGetInfo(&freeMem, &totalMem);
			//if (freeMem < init_alloc) // Couldn't find enough memory to do the initial alloc
			//    init_alloc = 1; // (0 is used as default parameter in cnmem)

			ctx->cnmem_device.size = init_alloc;
			cnmemDevice_t* devices = (cnmemDevice_t*) malloc(sizeof(cnmemDevice_t) * numDevices);
			memset(devices, 0, sizeof(cnmemDevice_t) * numDevices);
			for (int i = 0; i < numDevices; i++) {
				devices[i].device = _devices[i];
				devices[i].size = 1;
			}
			cnmemStatus_t cm_status = cnmemInit(numDevices, devices, CNMEM_FLAGS_DEFAULT);
			free(devices);
			if (cm_status != CNMEM_STATUS_SUCCESS)
				FatalError("Cannot initialize memory manager.", NVGRAPH_ERR_UNKNOWN);

			//Cublas and Cusparse
			nvgraph::Cusparse::get_handle();
			nvgraph::Cublas::get_handle();

			//others
			ctx->stream = 0;
			ctx->nvgraphIsInitialized = true;

			if (outCtx) {
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

			CHECK_CUDA(cudaFree((void * )0));
			CHECK_CUDA(cudaGetDevice(&device));
			struct nvgraphContext *ctx = NULL;
			ctx = (struct nvgraphContext *) malloc(sizeof(*ctx));
			if (!ctx) {
				FatalError("Cannot allocate NVGRAPH context.", NVGRAPH_ERR_UNKNOWN);
			}

			//cnmem
			memset(&ctx->cnmem_device, 0, sizeof(ctx->cnmem_device)); // init all to 0
			ctx->cnmem_device.device = device; // cnmem runs on the device set by cudaSetDevice

			size_t init_alloc = 1; // Initial allocation tentative, it is currently 1 so this feature is basically disabeled.

			// Warning : Should uncomment that if using init_alloc > 1
			//size_t freeMem, totalMem;
			//cudaMemGetInfo(&freeMem, &totalMem);
			//if (freeMem < init_alloc) // Couldn't find enough memory to do the initial alloc
			//    init_alloc = 1; // (0 is used as default parameter in cnmem)

			ctx->cnmem_device.size = init_alloc;

			cnmemStatus_t cm_status = cnmemInit(1, &ctx->cnmem_device, CNMEM_FLAGS_DEFAULT);
			if (cm_status != CNMEM_STATUS_SUCCESS)
				FatalError("Cannot initialize memory manager.", NVGRAPH_ERR_UNKNOWN);

			//Cublas and Cusparse
			nvgraph::Cusparse::get_handle();
			nvgraph::Cublas::get_handle();

			//others
			ctx->stream = 0;
			ctx->nvgraphIsInitialized = true;

			if (outCtx) {
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

			//Cublas and Cusparse
			nvgraph::Cusparse::destroy_handle();
			nvgraph::Cublas::destroy_handle();
			//cnmem

//     compiler is complaining, cm_status is not used in release build
#ifdef DEBUG
			cnmemStatus_t cm_status = cnmemFinalize();
			if( cm_status != CNMEM_STATUS_SUCCESS ) {
				CERR() << "Warning: " << cnmemGetErrorString(cm_status) << std::endl;
			}
#else
			cnmemFinalize();
#endif
			//others
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
			descrG = (struct nvgraphGraphDescr*) malloc(sizeof(*descrG));
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
				if (descrG->TT == NVGRAPH_2D_32I_32I) {
					switch (descrG->T) {
						case CUDA_R_32I: {
							nvgraph::Matrix2d<int32_t, int32_t, int32_t>* m =
									static_cast<nvgraph::Matrix2d<int32_t, int32_t, int32_t>*>(descrG->graph_handle);
							delete m;
							break;
						}
						default:
							return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
					}
				}
				else {
					switch (descrG->graphStatus) {
						case IS_EMPTY: {
							break;
						}
						case HAS_TOPOLOGY: {
							nvgraph::CsrGraph<int> *CSRG =
									static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
							delete CSRG;
							break;
						}
						case HAS_VALUES: {
							if (descrG->T == CUDA_R_32F) {
								nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
										static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
								delete MCSRG;
							}
							else if (descrG->T == CUDA_R_64F) {
								nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
										static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
								delete MCSRG;
							}
							else if (descrG->T == CUDA_R_32I) {
								nvgraph::MultiValuedCsrGraph<int, int> *MCSRG =
										static_cast<nvgraph::MultiValuedCsrGraph<int, int>*>(descrG->graph_handle);
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

	nvgraphStatus_t NVGRAPH_API nvgraphSetStream_impl(nvgraphHandle_t handle, cudaStream_t stream)
																		{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
			//CnMem
			cnmemStatus_t cm_status = cnmemRegisterStream(stream);
			if (cm_status != CNMEM_STATUS_SUCCESS)
				return NVGRAPH_STATUS_INTERNAL_ERROR;
			// nvgraph handle
			handle->stream = stream;
			//Cublas and Cusparse
			nvgraph::Cublas::setStream(stream);
			nvgraph::Cusparse::setStream(stream);
		}
		NVGRAPH_CATCHES(rc)

		return getCAPIStatusForError(rc);
	}

	nvgraphStatus_t NVGRAPH_API nvgraphSetGraphStructure_impl(nvgraphHandle_t handle,
																					nvgraphGraphDescr_t descrG,
																					void* topologyData,
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
						if (!t->nvertices || !t->nedges || check_ptr(t->source_offsets)
								|| check_ptr(t->destination_indices))
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
						if (!t->nvertices || !t->nedges || check_ptr(t->destination_offsets)
								|| check_ptr(t->source_indices))
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
				nvgraph::CsrGraph<int> * CSRG = new nvgraph::CsrGraph<int>(v, e, handle->stream);

				CHECK_CUDA(cudaMemcpy(CSRG->get_raw_row_offsets(),
												neighborhood,
												(size_t )((CSRG->get_num_vertices() + 1) * sizeof(int)),
												cudaMemcpyDefault));

				CHECK_CUDA(cudaMemcpy(CSRG->get_raw_column_indices(),
												edgedest,
												(size_t )((CSRG->get_num_edges()) * sizeof(int)),
												cudaMemcpyDefault));

				// Set the graph handle
				descrG->graph_handle = CSRG;
				descrG->graphStatus = HAS_TOPOLOGY;
			}
			else if (TT == NVGRAPH_2D_32I_32I) {
				nvgraph2dCOOTopology32I_t td = static_cast<nvgraph2dCOOTopology32I_t>(topologyData);
				switch (td->valueType) {
					case CUDA_R_32I: {
						if (!td->nvertices || !td->nedges || !td->source_indices
								|| !td->destination_indices || !td->numDevices || !td->devices
								|| !td->blockN)
							return NVGRAPH_STATUS_INVALID_VALUE;
						descrG->TT = TT;
						descrG->graphStatus = HAS_TOPOLOGY;
						if (td->values)
							descrG->graphStatus = HAS_VALUES;
						descrG->T = td->valueType;
						std::vector<int32_t> devices;
						for (int32_t i = 0; i < td->numDevices; i++)
							devices.push_back(td->devices[i]);
						nvgraph::MatrixDecompositionDescription<int32_t, int32_t> description(	td->nvertices,
																														td->blockN,
																														td->nedges,
																														devices);
						nvgraph::Matrix2d<int32_t, int32_t, int32_t>* m = new nvgraph::Matrix2d<int32_t,
								int32_t, int32_t>();
						*m = nvgraph::COOto2d(description,
														td->source_indices,
														td->destination_indices,
														(int32_t*) td->values);
						descrG->graph_handle = m;
						break;
					}
					default: {
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

	nvgraphStatus_t NVGRAPH_API nvgraphAttachGraphStructure_impl(nvgraphHandle_t handle,
															nvgraphGraphDescr_t descrG,
															void* topologyData,
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
						if (!t->nvertices || !t->nedges || check_ptr(t->source_offsets)
								|| check_ptr(t->destination_indices))
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
						if (!t->nvertices || !t->nedges || check_ptr(t->destination_offsets)
								|| check_ptr(t->source_indices))
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
				nvgraph::CsrGraph<int> * CSRG = new nvgraph::CsrGraph<int>(v, e, handle->stream);

				CSRG->set_raw_row_offsets(neighborhood);
				CSRG->set_raw_column_indices(edgedest);

				// Set the graph handle
				descrG->graph_handle = CSRG;
				descrG->graphStatus = HAS_TOPOLOGY;
			}
			else if (TT == NVGRAPH_2D_32I_32I) {
				nvgraph2dCOOTopology32I_t td = static_cast<nvgraph2dCOOTopology32I_t>(topologyData);
				switch (td->valueType) {
					case CUDA_R_32I: {
						if (!td->nvertices || !td->nedges || !td->source_indices
								|| !td->destination_indices || !td->numDevices || !td->devices
								|| !td->blockN)
							return NVGRAPH_STATUS_INVALID_VALUE;
						descrG->TT = TT;
						descrG->graphStatus = HAS_TOPOLOGY;
						if (td->values)
							descrG->graphStatus = HAS_VALUES;
						descrG->T = td->valueType;
						std::vector<int32_t> devices;
						for (int32_t i = 0; i < td->numDevices; i++)
							devices.push_back(td->devices[i]);
						nvgraph::MatrixDecompositionDescription<int32_t, int32_t> description(	td->nvertices,
																														td->blockN,
																														td->nedges,
																														devices);
						nvgraph::Matrix2d<int32_t, int32_t, int32_t>* m = new nvgraph::Matrix2d<int32_t,
								int32_t, int32_t>();
						*m = nvgraph::COOto2d(description,
														td->source_indices,
														td->destination_indices,
														(int32_t*) td->values);
						descrG->graph_handle = m;
						break;
					}
					default: {
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

	nvgraphStatus_t NVGRAPH_API nvgraphGetGraphStructure_impl(nvgraphHandle_t handle,
																					nvgraphGraphDescr_t descrG,
																					void* topologyData,
																					nvgraphTopologyType_t* TT)
																					{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle) || check_graph(descrG) || check_topology(descrG))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

			nvgraphTopologyType_t graphTType = descrG->TT;

			if (TT != NULL)
				*TT = graphTType;

			if (topologyData != NULL) {
				nvgraph::CsrGraph<int> *CSRG =
						static_cast<nvgraph::CsrGraph<int> *>(descrG->graph_handle);
				int v = static_cast<int>(CSRG->get_num_vertices());
				int e = static_cast<int>(CSRG->get_num_edges());
				int *neighborhood = NULL, *edgedest = NULL;

				switch (graphTType)
				{
					case NVGRAPH_CSR_32:
						{
						nvgraphCSRTopology32I_t t = static_cast<nvgraphCSRTopology32I_t>(topologyData);
						t->nvertices = static_cast<int>(v);
						t->nedges = static_cast<int>(e);
						neighborhood = t->source_offsets;
						edgedest = t->destination_indices;
						break;
					}
					case NVGRAPH_CSC_32:
						{
						nvgraphCSCTopology32I_t t = static_cast<nvgraphCSCTopology32I_t>(topologyData);
						t->nvertices = static_cast<int>(v);
						t->nedges = static_cast<int>(e);
						neighborhood = t->destination_offsets;
						edgedest = t->source_indices;
						break;
					}
					default:
						return NVGRAPH_STATUS_INTERNAL_ERROR;
				}

				if (neighborhood != NULL) {
					CHECK_CUDA(cudaMemcpy(neighborhood,
													CSRG->get_raw_row_offsets(),
													(size_t )((v + 1) * sizeof(int)),
													cudaMemcpyDefault));
				}

				if (edgedest != NULL) {
					CHECK_CUDA(cudaMemcpy(edgedest,
													CSRG->get_raw_column_indices(),
													(size_t )((e) * sizeof(int)),
													cudaMemcpyDefault));
				}

			}
		}
		NVGRAPH_CATCHES(rc)
		return getCAPIStatusForError(rc);
	}

	nvgraphStatus_t NVGRAPH_API nvgraphAllocateVertexData_impl(nvgraphHandle_t handle,
																					nvgraphGraphDescr_t descrG,
																					size_t numsets,
																					cudaDataType_t *settypes)
																					{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle) || check_graph(descrG) || check_int_size(numsets)
					|| check_ptr(settypes))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
			if (check_uniform_type_array(settypes, numsets))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (descrG->graphStatus == HAS_TOPOLOGY) // need to convert CsrGraph to MultiValuedCsrGraph first
					{
				if (*settypes == CUDA_R_32F)
						{
					nvgraph::CsrGraph<int> *CSRG =
							static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
					nvgraph::MultiValuedCsrGraph<int, float> *MCSRG = new nvgraph::MultiValuedCsrGraph<
							int, float>(*CSRG);
					descrG->graph_handle = MCSRG;
				}
				else if (*settypes == CUDA_R_64F)
						{
					nvgraph::CsrGraph<int> *CSRG =
							static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
					nvgraph::MultiValuedCsrGraph<int, double> *MCSRG = new nvgraph::MultiValuedCsrGraph<
							int, double>(*CSRG);
					descrG->graph_handle = MCSRG;
				}
				else if (*settypes == CUDA_R_32I)
						{
					nvgraph::CsrGraph<int> *CSRG =
							static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
					nvgraph::MultiValuedCsrGraph<int, int> *MCSRG = new nvgraph::MultiValuedCsrGraph<int,
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
			if (*settypes == CUDA_R_32F)
					{
				nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
				MCSRG->allocateVertexData(numsets, NULL);
			}
			else if (*settypes == CUDA_R_64F)
					{
				nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
				MCSRG->allocateVertexData(numsets, NULL);
			}
			else if (*settypes == CUDA_R_32I)
					{
				nvgraph::MultiValuedCsrGraph<int, int> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, int>*>(descrG->graph_handle);
				MCSRG->allocateVertexData(numsets, NULL);
			}
			else
				return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
		}
		NVGRAPH_CATCHES(rc)

		return getCAPIStatusForError(rc);
	}

	nvgraphStatus_t NVGRAPH_API nvgraphAttachVertexData_impl(nvgraphHandle_t handle,
															 nvgraphGraphDescr_t descrG,
															 size_t setnum,
															 cudaDataType_t settype,
															 void *vertexData)
															 {
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle) || check_graph(descrG) || check_int_size(setnum))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (descrG->graphStatus == HAS_TOPOLOGY) // need to convert CsrGraph to MultiValuedCsrGraph first
					{
				if (settype == CUDA_R_32F)
						{
					nvgraph::CsrGraph<int> *CSRG =
							static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
					nvgraph::MultiValuedCsrGraph<int, float> *MCSRG = new nvgraph::MultiValuedCsrGraph<
							int, float>(*CSRG);
					descrG->graph_handle = MCSRG;
				}
				else if (settype == CUDA_R_64F)
						{
					nvgraph::CsrGraph<int> *CSRG =
							static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
					nvgraph::MultiValuedCsrGraph<int, double> *MCSRG = new nvgraph::MultiValuedCsrGraph<
							int, double>(*CSRG);
					descrG->graph_handle = MCSRG;
				}
				else if (settype == CUDA_R_32I)
						{
					nvgraph::CsrGraph<int> *CSRG =
							static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
					nvgraph::MultiValuedCsrGraph<int, int> *MCSRG = new nvgraph::MultiValuedCsrGraph<int,
							int>(*CSRG);
					descrG->graph_handle = MCSRG;
				}
				else
					return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
				descrG->T = settype;
				descrG->graphStatus = HAS_VALUES;
			}
			else if (descrG->graphStatus == HAS_VALUES) // Already in MultiValuedCsrGraph, just need to check the type
					{
				if (settype != descrG->T)
					return NVGRAPH_STATUS_INVALID_VALUE;
			}
			else
				return NVGRAPH_STATUS_INVALID_VALUE;

			// transfer
			if (settype == CUDA_R_32F)
					{
				nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
				MCSRG->attachVertexData(setnum, (float*)vertexData, NULL);
			}
			else if (settype == CUDA_R_64F)
					{
				nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
				MCSRG->attachVertexData(setnum, (double*)vertexData, NULL);
			}
			else if (settype == CUDA_R_32I)
					{
				nvgraph::MultiValuedCsrGraph<int, int> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, int>*>(descrG->graph_handle);
				MCSRG->attachVertexData(setnum, (int*)vertexData, NULL);
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
																				cudaDataType_t *settypes)
																				{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle) || check_graph(descrG) || check_int_size(numsets)
					|| check_ptr(settypes))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
			if (check_uniform_type_array(settypes, numsets))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
			// Look at what kind of graph we have
			if (descrG->graphStatus == HAS_TOPOLOGY) // need to convert CsrGraph to MultiValuedCsrGraph first
					{
				if (*settypes == CUDA_R_32F)
						{
					nvgraph::CsrGraph<int> *CSRG =
							static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
					nvgraph::MultiValuedCsrGraph<int, float> *MCSRG = new nvgraph::MultiValuedCsrGraph<
							int, float>(*CSRG);
					descrG->graph_handle = MCSRG;
				}
				else if (*settypes == CUDA_R_64F)
						{
					nvgraph::CsrGraph<int> *CSRG =
							static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
					nvgraph::MultiValuedCsrGraph<int, double> *MCSRG = new nvgraph::MultiValuedCsrGraph<
							int, double>(*CSRG);
					descrG->graph_handle = MCSRG;
				}
				else if (*settypes == CUDA_R_32I)
						{
					nvgraph::CsrGraph<int> *CSRG =
							static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
					nvgraph::MultiValuedCsrGraph<int, int> *MCSRG = new nvgraph::MultiValuedCsrGraph<int,
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
			if (*settypes == CUDA_R_32F)
					{
				nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
				MCSRG->allocateEdgeData(numsets, NULL);
			}
			else if (*settypes == CUDA_R_64F)
					{
				nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
				MCSRG->allocateEdgeData(numsets, NULL);
			}
			else if (*settypes == CUDA_R_32I)
					{
				nvgraph::MultiValuedCsrGraph<int, int> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, int>*>(descrG->graph_handle);
				MCSRG->allocateEdgeData(numsets, NULL);
			}
			else
				return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;

		}
		NVGRAPH_CATCHES(rc)

		return getCAPIStatusForError(rc);
	}

	nvgraphStatus_t NVGRAPH_API nvgraphAttachEdgeData_impl(nvgraphHandle_t handle,
														   nvgraphGraphDescr_t descrG,
														   size_t setnum,
														   cudaDataType_t settype,
														   void *edgeData)
														   {
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle) || check_graph(descrG) || check_int_size(setnum))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
			// Look at what kind of graph we have
			if (descrG->graphStatus == HAS_TOPOLOGY) // need to convert CsrGraph to MultiValuedCsrGraph first
					{
				if (settype == CUDA_R_32F)
						{
					nvgraph::CsrGraph<int> *CSRG =
							static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
					nvgraph::MultiValuedCsrGraph<int, float> *MCSRG = new nvgraph::MultiValuedCsrGraph<
							int, float>(*CSRG);
					descrG->graph_handle = MCSRG;
				}
				else if (settype == CUDA_R_64F)
						{
					nvgraph::CsrGraph<int> *CSRG =
							static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
					nvgraph::MultiValuedCsrGraph<int, double> *MCSRG = new nvgraph::MultiValuedCsrGraph<
							int, double>(*CSRG);
					descrG->graph_handle = MCSRG;
				}
				else if (settype == CUDA_R_32I)
						{
					nvgraph::CsrGraph<int> *CSRG =
							static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
					nvgraph::MultiValuedCsrGraph<int, int> *MCSRG = new nvgraph::MultiValuedCsrGraph<int,
							int>(*CSRG);
					descrG->graph_handle = MCSRG;
				}
				else
					return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
				descrG->T = settype;
				descrG->graphStatus = HAS_VALUES;
			}
			else if (descrG->graphStatus == HAS_VALUES) // Already in MultiValuedCsrGraph, just need to check the type
					{
				if (settype != descrG->T)
					return NVGRAPH_STATUS_INVALID_VALUE;
			}
			else
				return NVGRAPH_STATUS_INVALID_VALUE;

			// Allocate and transfer
			if (settype == CUDA_R_32F)
					{
				nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
				MCSRG->attachEdgeData(setnum, (float*)edgeData, NULL);
			}
			else if (settype == CUDA_R_64F)
					{
				nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
				MCSRG->attachEdgeData(setnum, (double*)edgeData, NULL);
			}
			else if (settype == CUDA_R_32I)
					{
				nvgraph::MultiValuedCsrGraph<int, int> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, int>*>(descrG->graph_handle);
				MCSRG->attachEdgeData(setnum, (int*)edgeData, NULL);
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
			if (check_context(handle) || check_graph(descrG) || check_int_size(setnum)
					|| check_ptr(vertexData))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
				FatalError("Graph should have allocated values.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (descrG->T == CUDA_R_32F)
					{
				nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				cudaMemcpy(MCSRG->get_raw_vertex_dim(setnum),
								(float*) vertexData,
								(size_t) ((MCSRG->get_num_vertices()) * sizeof(float)),
								cudaMemcpyDefault);
			}
			else if (descrG->T == CUDA_R_64F)
					{
				nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				cudaMemcpy(MCSRG->get_raw_vertex_dim(setnum),
								(double*) vertexData,
								(size_t) ((MCSRG->get_num_vertices()) * sizeof(double)),
								cudaMemcpyDefault);
			}
			else if (descrG->T == CUDA_R_32I)
					{
				nvgraph::MultiValuedCsrGraph<int, int> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, int>*>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				cudaMemcpy(MCSRG->get_raw_vertex_dim(setnum),
								(int*) vertexData,
								(size_t) ((MCSRG->get_num_vertices()) * sizeof(int)),
								cudaMemcpyDefault);
			}
			else
				return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;

			cudaCheckError()
							;

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
			if (check_context(handle) || check_graph(descrG) || check_int_size(setnum)
					|| check_ptr(vertexData))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
				FatalError("Graph should have values.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (descrG->T == CUDA_R_32F)
					{
				nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				cudaMemcpy((float*) vertexData,
								MCSRG->get_raw_vertex_dim(setnum),
								(size_t) ((MCSRG->get_num_vertices()) * sizeof(float)),
								cudaMemcpyDefault);
			}
			else if (descrG->T == CUDA_R_64F)
					{
				nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				cudaMemcpy((double*) vertexData,
								MCSRG->get_raw_vertex_dim(setnum),
								(size_t) ((MCSRG->get_num_vertices()) * sizeof(double)),
								cudaMemcpyDefault);
			}
			else if (descrG->T == CUDA_R_32I)
					{
				nvgraph::MultiValuedCsrGraph<int, int> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, int>*>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				cudaMemcpy((int*) vertexData,
								MCSRG->get_raw_vertex_dim(setnum),
								(size_t) ((MCSRG->get_num_vertices()) * sizeof(int)),
								cudaMemcpyDefault);
			}
			else
				return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;

			cudaCheckError()
							;

		}
		NVGRAPH_CATCHES(rc)

		return getCAPIStatusForError(rc);
	}

	nvgraphStatus_t NVGRAPH_API nvgraphConvertTopology_impl(nvgraphHandle_t handle,
																				nvgraphTopologyType_t srcTType,
																				void *srcTopology,
																				void *srcEdgeData,
																				cudaDataType_t *dataType,
																				nvgraphTopologyType_t dstTType,
																				void *dstTopology,
																				void *dstEdgeData)
																				{

		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle) || check_ptr(dstEdgeData) || check_ptr(srcEdgeData))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

			size_t sizeT;
			if (*dataType == CUDA_R_32F)
				sizeT = sizeof(float);
			else if (*dataType == CUDA_R_64F)
				sizeT = sizeof(double);
			else
				return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;

			// Trust me, this better than nested if's.
			if (srcTType == NVGRAPH_CSR_32 && dstTType == NVGRAPH_CSR_32) {                  // CSR2CSR
				nvgraphCSRTopology32I_t srcT = static_cast<nvgraphCSRTopology32I_t>(srcTopology);
				nvgraphCSRTopology32I_t dstT = static_cast<nvgraphCSRTopology32I_t>(dstTopology);
				dstT->nvertices = srcT->nvertices;
				dstT->nedges = srcT->nedges;
				CHECK_CUDA(cudaMemcpy(dstT->source_offsets,
												srcT->source_offsets,
												(srcT->nvertices + 1) * sizeof(int),
												cudaMemcpyDefault));
				CHECK_CUDA(cudaMemcpy(dstT->destination_indices,
												srcT->destination_indices,
												srcT->nedges * sizeof(int),
												cudaMemcpyDefault));
				CHECK_CUDA(cudaMemcpy(dstEdgeData,
												srcEdgeData,
												srcT->nedges * sizeT,
												cudaMemcpyDefault));
			} else if (srcTType == NVGRAPH_CSR_32 && dstTType == NVGRAPH_CSC_32) {           // CSR2CSC
				nvgraphCSRTopology32I_t srcT = static_cast<nvgraphCSRTopology32I_t>(srcTopology);
				nvgraphCSCTopology32I_t dstT = static_cast<nvgraphCSCTopology32I_t>(dstTopology);
				dstT->nvertices = srcT->nvertices;
				dstT->nedges = srcT->nedges;
				csr2csc(srcT->nvertices, srcT->nvertices, srcT->nedges,
							srcEdgeData,
							srcT->source_offsets, srcT->destination_indices,
							dstEdgeData,
							dstT->source_indices, dstT->destination_offsets,
							CUSPARSE_ACTION_NUMERIC,
							CUSPARSE_INDEX_BASE_ZERO, dataType);
			} else if (srcTType == NVGRAPH_CSR_32 && dstTType == NVGRAPH_COO_32) {           // CSR2COO
				nvgraphCSRTopology32I_t srcT = static_cast<nvgraphCSRTopology32I_t>(srcTopology);
				nvgraphCOOTopology32I_t dstT = static_cast<nvgraphCOOTopology32I_t>(dstTopology);
				dstT->nvertices = srcT->nvertices;
				dstT->nedges = srcT->nedges;
				if (dstT->tag == NVGRAPH_SORTED_BY_SOURCE || dstT->tag == NVGRAPH_DEFAULT
						|| dstT->tag == NVGRAPH_UNSORTED) {
					csr2coo(srcT->source_offsets,
								srcT->nedges,
								srcT->nvertices,
								dstT->source_indices,
								CUSPARSE_INDEX_BASE_ZERO);
					CHECK_CUDA(cudaMemcpy(dstT->destination_indices,
													srcT->destination_indices,
													srcT->nedges * sizeof(int),
													cudaMemcpyDefault));
					CHECK_CUDA(cudaMemcpy(dstEdgeData,
													srcEdgeData,
													srcT->nedges * sizeT,
													cudaMemcpyDefault));
				} else if (dstT->tag == NVGRAPH_SORTED_BY_DESTINATION) {
					// Step 1: Convert to COO_Source
					csr2coo(srcT->source_offsets,
								srcT->nedges,
								srcT->nvertices,
								dstT->source_indices,
								CUSPARSE_INDEX_BASE_ZERO);
					// Step 2: Convert to COO_Destination
					cooSortByDestination(srcT->nvertices, srcT->nvertices, srcT->nedges,
												srcEdgeData,
												dstT->source_indices, srcT->destination_indices,
												dstEdgeData,
												dstT->source_indices, dstT->destination_indices,
												CUSPARSE_INDEX_BASE_ZERO,
												dataType);
				} else {
					return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
				}
				///////////////////////////////////////////////////////////////////////////////////////////////////////////
			} else if (srcTType == NVGRAPH_CSC_32 && dstTType == NVGRAPH_CSR_32) {           // CSC2CSR
				nvgraphCSCTopology32I_t srcT = static_cast<nvgraphCSCTopology32I_t>(srcTopology);
				nvgraphCSRTopology32I_t dstT = static_cast<nvgraphCSRTopology32I_t>(dstTopology);
				dstT->nvertices = srcT->nvertices;
				dstT->nedges = srcT->nedges;
				csc2csr(srcT->nvertices, srcT->nvertices, srcT->nedges,
							srcEdgeData,
							srcT->source_indices, srcT->destination_offsets,
							dstEdgeData,
							dstT->source_offsets, dstT->destination_indices,
							CUSPARSE_ACTION_NUMERIC,
							CUSPARSE_INDEX_BASE_ZERO, dataType);
			} else if (srcTType == NVGRAPH_CSC_32 && dstTType == NVGRAPH_CSC_32) {           // CSC2CSC
				nvgraphCSCTopology32I_t srcT = static_cast<nvgraphCSCTopology32I_t>(srcTopology);
				nvgraphCSCTopology32I_t dstT = static_cast<nvgraphCSCTopology32I_t>(dstTopology);
				dstT->nvertices = srcT->nvertices;
				dstT->nedges = srcT->nedges;
				CHECK_CUDA(cudaMemcpy(dstT->destination_offsets,
												srcT->destination_offsets,
												(srcT->nvertices + 1) * sizeof(int),
												cudaMemcpyDefault));
				CHECK_CUDA(cudaMemcpy(dstT->source_indices,
												srcT->source_indices,
												srcT->nedges * sizeof(int),
												cudaMemcpyDefault));
				CHECK_CUDA(cudaMemcpy(dstEdgeData,
												srcEdgeData,
												srcT->nedges * sizeT,
												cudaMemcpyDefault));
			} else if (srcTType == NVGRAPH_CSC_32 && dstTType == NVGRAPH_COO_32) {           // CSC2COO
				nvgraphCSCTopology32I_t srcT = static_cast<nvgraphCSCTopology32I_t>(srcTopology);
				nvgraphCOOTopology32I_t dstT = static_cast<nvgraphCOOTopology32I_t>(dstTopology);
				dstT->nvertices = srcT->nvertices;
				dstT->nedges = srcT->nedges;
				if (dstT->tag == NVGRAPH_SORTED_BY_SOURCE) {
					// Step 1: Convert to COO_Destination
					csr2coo(srcT->destination_offsets,
								srcT->nedges,
								srcT->nvertices,
								dstT->destination_indices,
								CUSPARSE_INDEX_BASE_ZERO);
					// Step 2: Convert to COO_Source
					cooSortBySource(srcT->nvertices, srcT->nvertices, srcT->nedges,
											srcEdgeData,
											srcT->source_indices, dstT->destination_indices,
											dstEdgeData,
											dstT->source_indices, dstT->destination_indices,
											CUSPARSE_INDEX_BASE_ZERO,
											dataType);
				} else if (dstT->tag == NVGRAPH_SORTED_BY_DESTINATION || dstT->tag == NVGRAPH_DEFAULT
						|| dstT->tag == NVGRAPH_UNSORTED) {
					csr2coo(srcT->destination_offsets,
								srcT->nedges,
								srcT->nvertices,
								dstT->destination_indices,
								CUSPARSE_INDEX_BASE_ZERO);
					CHECK_CUDA(cudaMemcpy(dstT->source_indices,
													srcT->source_indices,
													srcT->nedges * sizeof(int),
													cudaMemcpyDefault));
					CHECK_CUDA(cudaMemcpy(dstEdgeData,
													srcEdgeData,
													srcT->nedges * sizeT,
													cudaMemcpyDefault));
				} else {
					return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
				}
				///////////////////////////////////////////////////////////////////////////////////////////////////////////
			} else if (srcTType == NVGRAPH_COO_32 && dstTType == NVGRAPH_CSR_32) {           // COO2CSR
				nvgraphCOOTopology32I_t srcT = static_cast<nvgraphCOOTopology32I_t>(srcTopology);
				nvgraphCSRTopology32I_t dstT = static_cast<nvgraphCSRTopology32I_t>(dstTopology);
				dstT->nvertices = srcT->nvertices;
				dstT->nedges = srcT->nedges;
				if (srcT->tag == NVGRAPH_SORTED_BY_SOURCE) {
					coo2csr(srcT->source_indices,
								srcT->nedges,
								srcT->nvertices,
								dstT->source_offsets,
								CUSPARSE_INDEX_BASE_ZERO);
					CHECK_CUDA(cudaMemcpy(dstT->destination_indices,
													srcT->destination_indices,
													srcT->nedges * sizeof(int),
													cudaMemcpyDefault));
					CHECK_CUDA(cudaMemcpy(dstEdgeData,
													srcEdgeData,
													srcT->nedges * sizeT,
													cudaMemcpyDefault));
				} else if (srcT->tag == NVGRAPH_SORTED_BY_DESTINATION) {
					cood2csr(srcT->nvertices, srcT->nvertices, srcT->nedges,
								srcEdgeData,
								srcT->source_indices, srcT->destination_indices,
								dstEdgeData,
								dstT->source_offsets, dstT->destination_indices,
								CUSPARSE_INDEX_BASE_ZERO,
								dataType);
				} else if (srcT->tag == NVGRAPH_DEFAULT || srcT->tag == NVGRAPH_UNSORTED) {
					coou2csr(srcT->nvertices, srcT->nvertices, srcT->nedges,
								srcEdgeData,
								srcT->source_indices, srcT->destination_indices,
								dstEdgeData,
								dstT->source_offsets, dstT->destination_indices,
								CUSPARSE_INDEX_BASE_ZERO,
								dataType);
				} else {
					return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
				}
			} else if (srcTType == NVGRAPH_COO_32 && dstTType == NVGRAPH_CSC_32) {           // COO2CSC
				nvgraphCOOTopology32I_t srcT = static_cast<nvgraphCOOTopology32I_t>(srcTopology);
				nvgraphCSCTopology32I_t dstT = static_cast<nvgraphCSCTopology32I_t>(dstTopology);
				dstT->nvertices = srcT->nvertices;
				dstT->nedges = srcT->nedges;
				if (srcT->tag == NVGRAPH_SORTED_BY_SOURCE) {
					coos2csc(srcT->nvertices, srcT->nvertices, srcT->nedges,
								srcEdgeData,
								srcT->source_indices, srcT->destination_indices,
								dstEdgeData,
								dstT->source_indices, dstT->destination_offsets,
								CUSPARSE_INDEX_BASE_ZERO,
								dataType);
				} else if (srcT->tag == NVGRAPH_SORTED_BY_DESTINATION) {
					coo2csr(srcT->destination_indices,
								srcT->nedges,
								srcT->nvertices,
								dstT->destination_offsets,
								CUSPARSE_INDEX_BASE_ZERO);
					CHECK_CUDA(cudaMemcpy(dstT->source_indices,
													srcT->source_indices,
													srcT->nedges * sizeof(int),
													cudaMemcpyDefault));
					CHECK_CUDA(cudaMemcpy(dstEdgeData,
													srcEdgeData,
													srcT->nedges * sizeT,
													cudaMemcpyDefault));
				} else if (srcT->tag == NVGRAPH_DEFAULT || srcT->tag == NVGRAPH_UNSORTED) {
					coou2csc(srcT->nvertices, srcT->nvertices, srcT->nedges,
								srcEdgeData,
								srcT->source_indices, srcT->destination_indices,
								dstEdgeData,
								dstT->source_indices, dstT->destination_offsets,
								CUSPARSE_INDEX_BASE_ZERO,
								dataType);
				} else {
					return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
				}
			} else if (srcTType == NVGRAPH_COO_32 && dstTType == NVGRAPH_COO_32) {           // COO2COO
				nvgraphCOOTopology32I_t srcT = static_cast<nvgraphCOOTopology32I_t>(srcTopology);
				nvgraphCOOTopology32I_t dstT = static_cast<nvgraphCOOTopology32I_t>(dstTopology);
				dstT->nvertices = srcT->nvertices;
				dstT->nedges = srcT->nedges;
				if (srcT->tag == dstT->tag || dstT->tag == NVGRAPH_DEFAULT
						|| dstT->tag == NVGRAPH_UNSORTED) {
					CHECK_CUDA(cudaMemcpy(dstT->source_indices,
													srcT->source_indices,
													srcT->nedges * sizeof(int),
													cudaMemcpyDefault));
					CHECK_CUDA(cudaMemcpy(dstT->destination_indices,
													srcT->destination_indices,
													srcT->nedges * sizeof(int),
													cudaMemcpyDefault));
					CHECK_CUDA(cudaMemcpy(dstEdgeData,
													srcEdgeData,
													srcT->nedges * sizeT,
													cudaMemcpyDefault));
				} else if (dstT->tag == NVGRAPH_SORTED_BY_SOURCE) {
					cooSortBySource(srcT->nvertices, srcT->nvertices, srcT->nedges,
											srcEdgeData,
											srcT->source_indices, srcT->destination_indices,
											dstEdgeData,
											dstT->source_indices, dstT->destination_indices,
											CUSPARSE_INDEX_BASE_ZERO,
											dataType);
				} else if (dstT->tag == NVGRAPH_SORTED_BY_DESTINATION) {
					cooSortByDestination(srcT->nvertices, srcT->nvertices, srcT->nedges,
												srcEdgeData,
												srcT->source_indices, srcT->destination_indices,
												dstEdgeData,
												dstT->source_indices, dstT->destination_indices,
												CUSPARSE_INDEX_BASE_ZERO,
												dataType);
				} else {
					return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
				}

				///////////////////////////////////////////////////////////////////////////////////////////////////////////
			} else {
				return NVGRAPH_STATUS_INVALID_VALUE;
			}

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
			if (check_context(handle) || check_graph(descrG) || check_int_size(setnum)
					|| check_ptr(edgeData))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
				return NVGRAPH_STATUS_INVALID_VALUE;

			if (descrG->T == CUDA_R_32F)
					{
				nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_edge_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				cudaMemcpy(MCSRG->get_raw_edge_dim(setnum),
								(float*) edgeData,
								(size_t) ((MCSRG->get_num_edges()) * sizeof(float)),
								cudaMemcpyDefault);
			}
			else if (descrG->T == CUDA_R_64F)
					{
				nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_edge_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				cudaMemcpy(MCSRG->get_raw_edge_dim(setnum),
								(double*) edgeData,
								(size_t) ((MCSRG->get_num_edges()) * sizeof(double)),
								cudaMemcpyDefault);
			}
			else if (descrG->T == CUDA_R_32I)
					{
				nvgraph::MultiValuedCsrGraph<int, int> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, int>*>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_edge_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				cudaMemcpy(MCSRG->get_raw_edge_dim(setnum),
								(int*) edgeData,
								(size_t) ((MCSRG->get_num_edges()) * sizeof(int)),
								cudaMemcpyDefault);
			}
			else
				return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;

			cudaCheckError()
							;
		}
		NVGRAPH_CATCHES(rc)

		return getCAPIStatusForError(rc);
	}

	nvgraphStatus_t NVGRAPH_API nvgraphGetEdgeData_impl(nvgraphHandle_t handle,
																			nvgraphGraphDescr_t descrG,
																			void *edgeData,
																			size_t setnum)
																			{
		NVGRAPH_ERROR rc = NVGRAPH_OK;
		try
		{
			if (check_context(handle) || check_graph(descrG) || check_int_size(setnum)
					|| check_ptr(edgeData))
				FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

			if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
				return NVGRAPH_STATUS_INVALID_VALUE;

			if (descrG->T == CUDA_R_32F)
					{
				nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_edge_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				cudaMemcpy((float*) edgeData,
								MCSRG->get_raw_edge_dim(setnum),
								(size_t) ((MCSRG->get_num_edges()) * sizeof(float)),
								cudaMemcpyDefault);
			}
			else if (descrG->T == CUDA_R_64F)
					{
				nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
						static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
				if (setnum >= MCSRG->get_num_edge_dim()) // base index is 0
					return NVGRAPH_STATUS_INVALID_VALUE;
				cudaMemcpy((double*) edgeData,
								MCSRG->get_raw_edge_dim(setnum),
								(size_t) ((MCSRG->get_num_edges()) * sizeof(double)),
								cudaMemcpyDefault);
			}
			else
				return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;

			cudaCheckError()
							;

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
			if (check_context(handle) || check_graph(descrG) || check_int_size(weight_index)
					|| check_ptr(alpha))
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
				case CUDA_R_32F:
					{
					float alphaT = *static_cast<const float*>(alpha);
					if (alphaT <= 0.0f || alphaT >= 1.0f)
						return NVGRAPH_STATUS_INVALID_VALUE;
					nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
							static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
					if (weight_index >= MCSRG->get_num_edge_dim()
							|| bookmark >= MCSRG->get_num_vertex_dim()
							|| rank >= MCSRG->get_num_vertex_dim()) // base index is 0
						return NVGRAPH_STATUS_INVALID_VALUE;

					int n = static_cast<int>(MCSRG->get_num_vertices());
					nvgraph::Vector<float> guess(n, handle->stream);
					nvgraph::Vector<float> bm(n, handle->stream);
					if (has_guess)
						guess.copy(MCSRG->get_vertex_dim(rank));
					else
						guess.fill(static_cast<float>(1.0 / n));
					bm.copy(MCSRG->get_vertex_dim(bookmark));
					nvgraph::Pagerank<int, float> pagerank_solver(	*MCSRG->get_valued_csr_graph(weight_index),
																					bm);
					rc = pagerank_solver.solve(alphaT, guess, MCSRG->get_vertex_dim(rank), tol, max_it);
					break;
				}
				case CUDA_R_64F:
					{
					double alphaT = *static_cast<const double*>(alpha);
					if (alphaT <= 0.0 || alphaT >= 1.0)
						return NVGRAPH_STATUS_INVALID_VALUE;

					nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
							static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
					if (weight_index >= MCSRG->get_num_edge_dim()
							|| bookmark >= MCSRG->get_num_vertex_dim()
							|| rank >= MCSRG->get_num_vertex_dim()) // base index is 0
						return NVGRAPH_STATUS_INVALID_VALUE;

					int n = static_cast<int>(MCSRG->get_num_vertices());
					nvgraph::Vector<double> guess(n, handle->stream);
					nvgraph::Vector<double> bm(n, handle->stream);
					bm.copy(MCSRG->get_vertex_dim(bookmark));
					if (has_guess)
						guess.copy(MCSRG->get_vertex_dim(rank));
					else
						guess.fill(static_cast<float>(1.0 / n));
					nvgraph::Pagerank<int, double> pagerank_solver(	*MCSRG->get_valued_csr_graph(weight_index),
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
																			uint64_t* result)
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

			nvgraph::CsrGraph<int> *CSRG = static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
			if (CSRG == NULL)
				return NVGRAPH_STATUS_MAPPING_ERROR;
			nvgraph::triangles_counting::TrianglesCount<int> counter(*CSRG); /* stream, device */
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

nvgraphStatus_t NVGRAPH_API nvgraphGetProperty(libraryPropertyType type, int *value)
																{
	switch (type) {
		case MAJOR_VERSION:
			*value = CUDART_VERSION / 1000;
			break;
		case MINOR_VERSION:
			*value = (CUDART_VERSION % 1000) / 10;
			break;
		case PATCH_LEVEL:
			*value = 0;
			break;
		default:
			return NVGRAPH_STATUS_INVALID_VALUE;
	}
	return NVGRAPH_STATUS_SUCCESS;
}

nvgraphStatus_t NVGRAPH_API nvgraphCreate(nvgraphHandle_t *handle)
														{
	return nvgraph::nvgraphCreate_impl(handle);
}

nvgraphStatus_t NVGRAPH_API nvgraphCreateMulti(nvgraphHandle_t *handle,
																int numDevices,
																int* devices) {
	return nvgraph::nvgraphCreateMulti_impl(handle, numDevices, devices);
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

nvgraphStatus_t NVGRAPH_API nvgraphSetStream(nvgraphHandle_t handle, cudaStream_t stream)
															{
	return nvgraph::nvgraphSetStream_impl(handle, stream);
}

nvgraphStatus_t NVGRAPH_API nvgraphSetGraphStructure(nvgraphHandle_t handle,
																		nvgraphGraphDescr_t descrG,
																		void* topologyData,
																		nvgraphTopologyType_t topologyType)
																		{
	return nvgraph::nvgraphSetGraphStructure_impl(handle, descrG, topologyData, topologyType);
}
nvgraphStatus_t NVGRAPH_API nvgraphGetGraphStructure(nvgraphHandle_t handle,
																		nvgraphGraphDescr_t descrG,
																		void* topologyData,
																		nvgraphTopologyType_t* topologyType)
																		{
	return nvgraph::nvgraphGetGraphStructure_impl(handle, descrG, topologyData, topologyType);
}
nvgraphStatus_t NVGRAPH_API nvgraphAllocateVertexData(nvgraphHandle_t handle,
																		nvgraphGraphDescr_t descrG,
																		size_t numsets,
																		cudaDataType_t *settypes)
																		{
	return nvgraph::nvgraphAllocateVertexData_impl(handle, descrG, numsets, settypes);
}

nvgraphStatus_t NVGRAPH_API nvgraphAllocateEdgeData(nvgraphHandle_t handle,
																		nvgraphGraphDescr_t descrG,
																		size_t numsets,
																		cudaDataType_t *settypes)
																		{
	return nvgraph::nvgraphAllocateEdgeData_impl(handle, descrG, numsets, settypes);
}

nvgraphStatus_t NVGRAPH_API nvgraphExtractSubgraphByVertex(nvgraphHandle_t handle,
																				nvgraphGraphDescr_t descrG,
																				nvgraphGraphDescr_t subdescrG,
																				int *subvertices,
																				size_t numvertices)
																				{
	return nvgraph::nvgraphExtractSubgraphByVertex_impl(handle,
																			descrG,
																			subdescrG,
																			subvertices,
																			numvertices);
}

nvgraphStatus_t NVGRAPH_API nvgraphExtractSubgraphByEdge(nvgraphHandle_t handle,
																			nvgraphGraphDescr_t descrG,
																			nvgraphGraphDescr_t subdescrG,
																			int *subedges,
																			size_t numedges)
																			{
	return nvgraph::nvgraphExtractSubgraphByEdge_impl(handle, descrG, subdescrG, subedges, numedges);
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

nvgraphStatus_t NVGRAPH_API nvgraphConvertTopology(nvgraphHandle_t handle,
																	nvgraphTopologyType_t srcTType,
																	void *srcTopology,
																	void *srcEdgeData,
																	cudaDataType_t *dataType,
																	nvgraphTopologyType_t dstTType,
																	void *dstTopology,
																	void *dstEdgeData) {
	return nvgraph::nvgraphConvertTopology_impl(handle,
																srcTType,
																srcTopology,
																srcEdgeData,
																dataType,
																dstTType,
																dstTopology,
																dstEdgeData);
}

nvgraphStatus_t NVGRAPH_API nvgraphSetEdgeData(nvgraphHandle_t handle,
																nvgraphGraphDescr_t descrG,
																void *edgeData,
																size_t setnum) {
	return nvgraph::nvgraphSetEdgeData_impl(handle, descrG, edgeData, setnum);
}

nvgraphStatus_t NVGRAPH_API nvgraphGetEdgeData(nvgraphHandle_t handle,
																nvgraphGraphDescr_t descrG,
																void *edgeData,
																size_t setnum) {
	return nvgraph::nvgraphGetEdgeData_impl(handle, descrG, edgeData, setnum);
}

nvgraphStatus_t NVGRAPH_API nvgraphSrSpmv(nvgraphHandle_t handle,
														const nvgraphGraphDescr_t descrG,
														const size_t weight_index,
														const void *alpha,
														const size_t x,
														const void *beta,
														const size_t y,
														const nvgraphSemiring_t SR) {
	return nvgraph::nvgraphSrSpmv_impl_cub(handle, descrG, weight_index, alpha, x, beta, y, SR);
}

nvgraphStatus_t NVGRAPH_API nvgraphSssp(nvgraphHandle_t handle,
														const nvgraphGraphDescr_t descrG,
														const size_t weight_index,
														const int *source_vert,
														const size_t sssp) {
	return nvgraph::nvgraphSssp_impl(handle, descrG, weight_index, source_vert, sssp);
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
																	uint64_t* result)
																	{
	return nvgraph::nvgraphTriangleCount_impl(handle, descrG, result);
}


nvgraphStatus_t NVGRAPH_API nvgraphAttachGraphStructure(nvgraphHandle_t handle,
														nvgraphGraphDescr_t descrG,
														void* topologyData,
														nvgraphTopologyType_t TT) {
	return nvgraph::nvgraphAttachGraphStructure_impl( handle, descrG, topologyData, TT);
}

nvgraphStatus_t NVGRAPH_API nvgraphAttachVertexData(nvgraphHandle_t handle,
													 nvgraphGraphDescr_t descrG,
													 size_t setnum,
													 cudaDataType_t settype,
													 void *vertexData) {
	return nvgraph::nvgraphAttachVertexData_impl( handle, descrG, setnum, settype, vertexData);
}

nvgraphStatus_t NVGRAPH_API nvgraphAttachEdgeData(nvgraphHandle_t handle,
											      nvgraphGraphDescr_t descrG,
											      size_t setnum,
											      cudaDataType_t settype,
											      void *edgeData) {
	return nvgraph::nvgraphAttachEdgeData_impl( handle, descrG, setnum, settype, edgeData);
}

