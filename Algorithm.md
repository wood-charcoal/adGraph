# Algorithm

## Page Rank

### page rank solver

- head files

```cpp
// ==========  page_rank_kernel.hpp ========== 
namespace nvgraph
{	
	template <typename ValueType_>
    void update_dangling_nodes(int n, ValueType_* dangling_nodes, ValueType_ damping_factor,  hipStream_t stream = 0);

} // end namespace nvgraph

// ==========  page_rank.hpp ========== 
namespace nvgraph
{
template <typename IndexType_, typename ValueType_>
class Pagerank 
{
public: 
    typedef IndexType_ IndexType;
    typedef ValueType_ ValueType;

private:
	ValuedCsrGraph <IndexType, ValueType> m_network ;
	Vector <ValueType> m_a;
	Vector <ValueType> m_b;
	Vector <ValueType> m_pagerank;
	Vector <ValueType> m_tmp;
	ValueType m_damping_factor;
	ValueType m_residual;
	ValueType m_tolerance;
	hipStream_t m_stream;
	int m_iterations;
	int m_max_it;
	bool m_is_setup;
	bool m_has_guess;

	bool solve_it();
	//void update_dangling_nodes(Vector<ValueType_>& dangling_nodes);
	void setup(ValueType damping_factor, Vector<ValueType>& initial_guess, Vector<ValueType>& pagerank_vector);

public:
	// Simple constructor 
	Pagerank(void) {};
	// Simple destructor
	~Pagerank(void) {};

	// Create a Pagerank Solver attached to a the transposed of a transition matrix
	// *** network is the transposed of a transition matrix***
	Pagerank(const ValuedCsrGraph <IndexType, ValueType>& network, Vector<ValueType>& dangling_nodes, hipStream_t stream = 0);
	
	// dangling_nodes is a vector of size n where dangling_nodes[i] = 1.0 if vertex i is a dangling node and 0.0 otherwise
    // pagerank_vector is the output
    //void solve(ValueType damping_factor, Vector<ValueType>& dangling_nodes, Vector<ValueType>& pagerank_vector);
   // setup with an initial guess of the pagerank
    NVGRAPH_ERROR solve(ValueType damping_factor, Vector<ValueType>& initial_guess, Vector<ValueType>& pagerank_vector, float tolerance =1.0E-6, int max_it = 500);
    inline ValueType get_residual() const {return m_residual;}
    inline int get_iterations() const {return m_iterations;}


// init :
// We need the transpose (=converse =reverse) in input (this can be seen as a CSC matrix that we see as CSR)
// b is a constant and uniform vector, b = 1.0/num_vertices
// a is a constant vector that initialy store the dangling nodes then we set : a = alpha*a + (1-alpha)e
// pagerank is 0
// tmp is random ( 1/n is fine)
// alpha is a constant scalar (0.85 usually)

//loop :
//  pagerank = csrmv (network, tmp)
//  scal(pagerank, alpha); //pagerank =  alpha*pagerank
//  gamma  = dot(a, tmp); //gamma  = a*tmp
//  pagerank = axpy(b, pagerank, gamma); // pagerank = pagerank+gamma*b

// convergence check
//  tmp = axpby(pagerank, tmp, -1, 1);	 // tmp = pagerank - tmp
//  residual_norm = norm(tmp);               
//  if converged (residual_norm)
	  // l1 = l1_norm(pagerank);
	  // pagerank = scal(pagerank, 1/l1);
      // return pagerank 
//  swap(tmp, pagerank)
//end loop
};

} // end namespace nvgraph
```

- source files

```cpp
// ==========  page_rank_kernel.cpp ========== 
namespace nvgraph
{

template <typename ValueType_>
__global__ void update_dn_kernel(int num_vertices, ValueType_* aa, ValueType_ beta)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int r = tidx; r < num_vertices; r += blockDim.x * gridDim.x)
    {
        // NOTE 1 : a = alpha*a + (1-alpha)e
        if (aa[r] == 0.0)
            aa[r] = beta; // NOTE 2 : alpha*0 + (1-alpha)*1 = (1-alpha)
    }
}

template <typename ValueType_>
void update_dangling_nodes(int num_vertices, ValueType_* dangling_nodes, ValueType_ damping_factor, hipStream_t stream)
{
	
	int num_threads = 256;
    int max_grid_size = 4096;
    int num_blocks = std::min(max_grid_size, (num_vertices/num_threads)+1);
    ValueType_ beta = 1.0-damping_factor;
    update_dn_kernel<<<num_blocks, num_threads, 0, stream>>>(num_vertices, dangling_nodes,beta);
    cudaCheckError();
}

//Explicit

template void update_dangling_nodes<double> (int num_vertices, double* dangling_nodes, double damping_factor, hipStream_t stream);
template void update_dangling_nodes<float> (int num_vertices, float* dangling_nodes, float damping_factor, hipStream_t stream);
} // end namespace nvgraph


// ==========  page_rank.cpp ========== 
namespace nvgraph
{
template <typename IndexType_, typename ValueType_>
Pagerank<IndexType_, ValueType_>::Pagerank(const ValuedCsrGraph <IndexType, ValueType>& network, Vector<ValueType>& dangling_nodes, hipStream_t stream)
    :m_network(network), m_a(dangling_nodes), m_stream(stream)
{
    // initialize cuda libs outside of the solve (this is slow)
    Cusparse::get_handle();
    Cublas::get_handle();
    m_residual = 1000.0;
    m_damping_factor = 0.0;
}

template <typename IndexType_, typename ValueType_>
void Pagerank<IndexType_, ValueType_>::setup(ValueType damping_factor, Vector<ValueType>& initial_guess, Vector<ValueType>& pagerank_vector)
{
    int n = static_cast<int>(m_network.get_num_vertices());
//    int nnz = static_cast<int>(m_network.get_num_edges());
#ifdef DEBUG
    if (n != static_cast<int>(initial_guess.get_size()) || n != static_cast<int>(m_a.get_size()) || n != static_cast<int>(pagerank_vector.get_size()))
    {
        CERR() << "n : " << n << std::endl;
        CERR() << "m_network.get_num_edges() " << m_network.get_num_edges() << std::endl;
        CERR() << "m_a : " << m_a.get_size() << std::endl;
        CERR() << "initial_guess.get_size() : " << initial_guess.get_size() << std::endl;
        CERR() << "pagerank_vector.get_size() : " << pagerank_vector.get_size() << std::endl;
        FatalError("Wrong input vector in Pagerank solver.", NVGRAPH_ERR_BAD_PARAMETERS);
    }
#endif
    if (damping_factor > 0.999 || damping_factor < 0.0001)
        FatalError("Wrong damping factor value in Pagerank solver.", NVGRAPH_ERR_BAD_PARAMETERS);
	m_damping_factor = damping_factor;
    m_tmp = initial_guess;
    m_pagerank = pagerank_vector;
    //dump(m_a.raw(), 100, 0);
	update_dangling_nodes(n, m_a.raw(), this->m_damping_factor, m_stream);
    //dump(m_a.raw(), 100, 0);
	m_b.allocate(n, m_stream);
    //m_b.dump(0,n);
    ValueType_ val =  static_cast<ValueType_>( 1.0/n);

    //fill_raw_vec(m_b.raw(), n, val); 
    // auto b = m_b.raw();
     m_b.fill(val, m_stream);
    // WARNING force initialization of the initial guess
    //fill(m_tmp.raw(), n, 1.1); 
}

template <typename IndexType_, typename ValueType_>
bool Pagerank<IndexType_, ValueType_>::solve_it()
{
	
    int n = static_cast<int>(m_network.get_num_vertices()), nnz = static_cast<int>(m_network.get_num_edges());
    int inc = 1;
    ValueType_  dot_res;

    ValueType *a = m_a.raw(),
         *b = m_b.raw(),
         *pr = m_pagerank.raw(),
         *tmp = m_tmp.raw();
    
    // normalize the input vector (tmp)
    if(m_iterations == 0)
        Cublas::scal(n, (ValueType_)1.0/Cublas::nrm2(n, tmp, inc) , tmp, inc);
    
    //spmv : pr = network * tmp
#ifdef NEW_CSRMV
    ValueType_ alpha = cub_semiring::hipcub::PlusTimesSemiring<ValueType_>::times_ident(); // 1.
    ValueType_ beta = cub_semiring::hipcub::PlusTimesSemiring<ValueType_>::times_null(); // 0.
    SemiringDispatch<IndexType_, ValueType_>::template Dispatch< cub_semiring::hipcub::PlusTimesSemiring<ValueType_> >(
        m_network.get_raw_values(),
        m_network.get_raw_row_offsets(),
        m_network.get_raw_column_indices(),
        tmp,
        pr,
        alpha,
        beta, 
        n,
        n,
        nnz,
        m_stream);
#else
    ValueType_  alpha = 1.0, beta =0.0;
#if __cplusplus > 199711L
    Semiring SR = Semiring::PlusTimes;
#else
    Semiring SR = PlusTimes;
#endif
    csrmv_mp<IndexType_, ValueType_>(n, n, nnz, 
           alpha,
           m_network,
           tmp,
           beta,
           pr,
           SR, 
           m_stream);
#endif
    
    // Rank one updates
    Cublas::scal(n, m_damping_factor, pr, inc);
    Cublas::dot(n, a, inc, tmp, inc, &dot_res);
    Cublas::axpy(n, dot_res, b, inc, pr, inc);

    // CVG check
    // we need to normalize pr to compare it to tmp 
    // (tmp has been normalized and overwitted at the beginning)
    Cublas::scal(n, (ValueType_)1.0/Cublas::nrm2(n, pr, inc) , pr, inc);
    
    // v = v - x
    Cublas::axpy(n, (ValueType_)-1.0, pr, inc, tmp, inc);
    m_residual = Cublas::nrm2(n, tmp, inc);

    if (m_residual < m_tolerance) // We know lambda = 1 for Pagerank
    {
        // CONVERGED
        // WARNING Norm L1 is more standard for the output of PageRank
        //m_pagerank.dump(0,m_pagerank.get_size());
        Cublas::scal(m_pagerank.get_size(), (ValueType_)1.0/m_pagerank.nrm1(m_stream), pr, inc);
        return true;
    }
    else
    {
        // m_pagerank.dump(0,m_pagerank.get_size());
        std::swap(m_pagerank, m_tmp);
        return false;
    }
}

template <typename IndexType_, typename ValueType_>
NVGRAPH_ERROR Pagerank<IndexType_, ValueType_>::solve(ValueType damping_factor, Vector<ValueType>& initial_guess, Vector<ValueType>& pagerank_vector, float tolerance, int max_it)
{
   
    #ifdef PR_VERBOSE
        std::stringstream ss;
        ss.str(std::string());
        size_t used_mem, free_mem, total_mem;
        ss <<" ------------------PageRank------------------"<< std::endl;
        ss <<" --------------------------------------------"<< std::endl;
        ss << std::setw(10) << "Iteration" << std::setw(20) << " Mem Usage (MB)" << std::setw(15) << "Residual" << std::endl;
        ss <<" --------------------------------------------"<< std::endl;
        COUT()<<ss.str();
        cuda_timer timer; timer.start();
    #endif
    m_max_it = max_it;
    m_tolerance = static_cast<ValueType_>(tolerance);
    setup(damping_factor, initial_guess, pagerank_vector);
    bool converged = false;
    int i = 0;

    while (!converged && i < m_max_it)
    { 
        m_iterations = i;
        converged = solve_it();
        i++;
         #ifdef PR_VERBOSE
            ss.str(std::string());
            cnmemMemGetInfo(&free_mem, &total_mem, NULL);
            used_mem=total_mem-free_mem;
            ss << std::setw(10) << i ;
            ss.precision(3);
            ss << std::setw(20) << std::fixed << used_mem/1024.0/1024.0;
            ss << std::setw(15) << std::scientific << m_residual  << std::endl;
            COUT()<<ss.str();
        #endif
    }
    m_iterations = i;
    #ifdef PR_VERBOSE
        COUT() <<" --------------------------------------------"<< std::endl;
        //stop timer
        COUT() <<" Total Time : "<< timer.stop() << "ms"<<std::endl;
        COUT() <<" --------------------------------------------"<< std::endl;
    #endif
    
    if (converged)    
    {
        pagerank_vector = m_pagerank;
    }
    else
    {
        // still return something even if we didn't converged 
        Cublas::scal(m_pagerank.get_size(), (ValueType_)1.0/m_tmp.nrm1(m_stream), m_tmp.raw(), 1);
        pagerank_vector = m_tmp;
    }
        //m_pagerank.dump(0,m_pagerank.get_size());
        //pagerank_vector.dump(0,pagerank_vector.get_size());
    return converged ? NVGRAPH_OK : NVGRAPH_ERR_NOT_CONVERGED;
}

template class Pagerank<int, double>;
template class Pagerank<int, float>;

// init :
// We actually need the transpose (=converse =reverse) of the original network, if the inuput is the original network then we have to transopose it	
// b is a constant and uniform vector, b = 1.0/num_vertices
// a is a constant vector that initialy store the dangling nodes then we set : a = alpha*a + (1-alpha)e
// pagerank is 0 
// tmp is random
// alpha is a constant scalar (0.85 usually)

//loop :
//  pagerank = csrmv (network, tmp)
//  scal(pagerank, alpha); //pagerank =  alpha*pagerank
//  gamma  = dot(a, tmp); //gamma  = a*tmp
//  pagerank = axpy(b, pagerank, gamma); // pagerank = pagerank+gamma*b

// convergence check
//  tmp = axpby(pagerank, tmp, -1, 1);	 // tmp = pagerank - tmp
//  residual_norm = norm(tmp);               
//  if converged (residual_norm)
	  // l1 = l1_norm(pagerank);
	  // pagerank = scal(pagerank, 1/l1);
      // return pagerank 
//  swap(tmp, pagerank)
//end loop

} // end namespace nvgraph
```

### api call

```cpp
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
```

## Triangle Count

### triangle count solver

- head files

```cpp
// ==========  triangle_count_kernel.hpp ========== 
namespace nvgraph
{

namespace triangles_counting
{

template <typename T>
void tricnt_thr(T nblock, spmat_t<T> *m, uint64_t *ocnt_d, ihipStream_t* stream);

template <typename T>
uint64_t reduce(uint64_t *v_d, T n, ihipStream_t* stream);
template <typename T>
void create_nondangling_vector(const T *roff, T *p_nonempty, T *n_nonempty, size_t n, ihipStream_t* stream);
} // namespace triangles_counting

} // namespace nvgraph

// ========== triangle_count.hpp ==========
namespace nvgraph
{

namespace triangles_counting
{


typedef enum { TCOUNT_THR } TrianglesCountAlgo;


template <typename IndexType>
class TrianglesCount 
{
private:
    //CsrGraph <IndexType>& m_last_graph ;
    AsyncEvent          m_event;
    uint64_t            m_triangles_number;
    spmat_t<IndexType>  m_mat;
    int                 m_dev_id;
    hipDeviceProp_t      m_dev_props;

    Vector<IndexType>   m_seq;

    hipStream_t        m_stream;

    bool m_done;

    void tcount_thr();

public:
    // Simple constructor 
    TrianglesCount(const CsrGraph <IndexType>& graph, hipStream_t stream = NULL, int device_id = -1);
    // Simple destructor
    ~TrianglesCount();

    NVGRAPH_ERROR count(TrianglesCountAlgo algo = TCOUNT_THR ); // TCOUNT_DEFAULT );
    inline uint64_t get_triangles_count() const {return m_triangles_number;}
};

} // end namespace triangles_counting

} // end namespace nvgraph
```

- source files

1. triangle_counting_kernel.cpp

```cpp
template<typename T>
__device__  __forceinline__ T LDG(const T* x)
                                 {
#if __CUDA_ARCH__ < 350
  return *x;
#else
  return __ldg(x);
#endif
}

namespace nvgraph
{

  namespace triangles_counting
  {

// hide behind 
    void* tmp_get(size_t size, ihipStream_t* stream)
                  {
      void *t = NULL;
      cnmemStatus_t status = cnmemMalloc(&t, size, stream);
      if (status == CNMEM_STATUS_OUT_OF_MEMORY)
          {
        FatalError("Not enough memory", NVGRAPH_ERR_NO_MEMORY);
      }
      else if (status != CNMEM_STATUS_SUCCESS)
          {
        FatalError("Memory manager internal error (alloc)", NVGRAPH_ERR_UNKNOWN);
      }

      return t;
    }

    void tmp_release(void* ptr, ihipStream_t* stream)
                     {
      cnmemStatus_t status = cnmemFree(ptr, stream);
      if (status != CNMEM_STATUS_SUCCESS)
          {
        FatalError("Memory manager internal error (release)", NVGRAPH_ERR_UNKNOWN);
      }
    }

// cub utility wrappers ////////////////////////////////////////////////////////
    template<typename InputIteratorT, typename OutputIteratorT>
    static inline void cubSum(InputIteratorT d_in, OutputIteratorT d_out,
                              int num_items,
                              ihipStream_t* stream = 0,
                              bool debug_synchronous = false) {

      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;

      hipcub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                             d_in,
                             d_out, num_items, stream,
                             debug_synchronous);
      cudaCheckError()
      ;
      d_temp_storage = tmp_get(temp_storage_bytes, stream);
      hipcub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                             d_in,
                             d_out, num_items, stream,
                             debug_synchronous);
      cudaCheckError()
      ;
      tmp_release(d_temp_storage, stream);

      return;
    }

    template<typename InputIteratorT,
        typename OutputIteratorT,
        typename NumSelectedIteratorT,
        typename SelectOp>
    static inline void cubIf(InputIteratorT d_in, OutputIteratorT d_out,
                             NumSelectedIteratorT d_num_selected_out,
                             int num_items, SelectOp select_op,
                             ihipStream_t* stream = 0,
                             bool debug_synchronous = false) {

      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;

      hipcub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                            d_in,
                            d_out, d_num_selected_out,
                            num_items,
                            select_op, stream,
                            debug_synchronous);
      cudaCheckError()
      ;
      d_temp_storage = tmp_get(temp_storage_bytes, stream);
      hipcub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                            d_in,
                            d_out, d_num_selected_out,
                            num_items,
                            select_op, stream,
                            debug_synchronous);
      cudaCheckError()
      ;
      tmp_release(d_temp_storage, stream);

      return;
    }

    template<typename T2>
    __device__ __host__ inline bool operator==(const T2 &lhs, const T2 &rhs) {
      return (lhs.x == rhs.x && lhs.y == rhs.y);
    }

//////////////////////////////////////////////////////////////////////////////////////////
    template<int BDIM_X,
        int BDIM_Y,
        int WSIZE,
        typename T>
    __device__  __forceinline__ T block_sum(T v) {

      __shared__ T sh[BDIM_X * BDIM_Y / WSIZE];

      const int lid = threadIdx.x % 32;
      const int wid = threadIdx.x / 32 + ((BDIM_Y > 1) ? threadIdx.y * (BDIM_X / 32) : 0);

      #pragma unroll
      for (int i = WSIZE / 2; i; i >>= 1) {
        v += utils::shfl_down(v, i);
      }
      if (lid == 0)
        sh[wid] = v;

      __syncthreads();
      if (wid == 0) {
        v = (lid < (BDIM_X * BDIM_Y / WSIZE)) ? sh[lid] : 0;

        #pragma unroll
        for (int i = (BDIM_X * BDIM_Y / WSIZE) / 2; i; i >>= 1) {
          v += utils::shfl_down(v, i);
        }
      }
      return v;
    }

//////////////////////////////////////////////////////////////////////////////////////////
    template<int BDIM,
        int LOCLEN,
        typename ROW_T,
        typename OFF_T,
        typename CNT_T>
    __global__ void tricnt_thr_k(const ROW_T ner,
                                 const ROW_T *__restrict__ rows,
                                 const OFF_T *__restrict__ roff,
                                 const ROW_T *__restrict__ cols,
                                 CNT_T *__restrict__ ocnt) {
      CNT_T __cnt = 0;
      const ROW_T tid = blockIdx.x * BDIM + threadIdx.x;

      for (ROW_T rid = tid; rid < ner; rid += gridDim.x * BDIM) {

        const ROW_T r = rows[rid];

        const OFF_T rbeg = roff[r];
        const OFF_T rend = roff[r + 1];
        const ROW_T rlen = rend - rbeg;

        if (!rlen)
          continue;
        if (rlen <= LOCLEN) {
          int nloc = 0;
          ROW_T loc[LOCLEN];

#pragma unroll
          for (nloc = 0; nloc < LOCLEN; nloc++) {
            if (rbeg + nloc >= rend)
              break;
            loc[nloc] = LDG(cols + rbeg + nloc);
          }

#pragma unroll
          for (int i = 1; i < LOCLEN; i++) {

            if (i == nloc)
              break;

            const ROW_T c = loc[i];
            const OFF_T soff = roff[c];
            const OFF_T eoff = roff[c + 1];

            for (OFF_T k = eoff - 1; k >= soff; k--) {

              const ROW_T cc = LDG(cols + k);
              if (cc < loc[0])
                break;

              for (int j = i - 1; j >= 0; j--) {
                if (cc == loc[j])
                  __cnt++;
              }
            }
          }
        } else {
          const ROW_T minc = cols[rbeg];
          for (int i = 1; i < rlen; i++) {

            const ROW_T c = LDG(cols + rbeg + i);
            const OFF_T soff = roff[c];
            const OFF_T eoff = roff[c + 1];

            for (OFF_T k = eoff - 1; k >= soff; k--) {

              const ROW_T cc = LDG(cols + k);
              if (cc < minc)
                break;

              for (int j = i - 1; j >= 0; j--) {
                if (cc == LDG(cols + rbeg + j))
                  __cnt++;
              }
            }
          }
        }
      }

      __syncthreads();
      __cnt = block_sum<BDIM, 1, 32>(__cnt);
      if (threadIdx.x == 0)
        ocnt[blockIdx.x] = __cnt;

      return;
    }

    template<typename T>
    void tricnt_thr(T nblock, spmat_t<T> *m, uint64_t *ocnt_d, ihipStream_t* stream) {

      hipFuncSetCacheConfig(reinterpret_cast<const void*>(tricnt_thr_k<THREADS, TH_CENT_K_LOCLEN, typename type_utils<T>::LOCINT, typename type_utils<T>::LOCINT, uint64_t>), hipFuncCachePreferL1);

      tricnt_thr_k<THREADS, TH_CENT_K_LOCLEN> <<<nblock, THREADS, 0, stream>>>(m->nrows, m->rows_d,
                                                                               m->roff_d,
                                                                               m->cols_d,
                                                                               ocnt_d);
      cudaCheckError()
      ;
      return;
    }

/////////////////////////////////////////////////////////////////
    template<typename IndexType>
    struct NonEmptyRow
    {
      const IndexType* p_roff;
      __host__ __device__ NonEmptyRow(const IndexType* roff) :
          p_roff(roff) {
      }
      __host__ __device__ __forceinline__
      bool operator()(const IndexType &a) const
                      {
        return (p_roff[a] < p_roff[a + 1]);
      }
    };

    template<typename T>
    void create_nondangling_vector(const T* roff,
                                   T *p_nonempty,
                                   T *n_nonempty,
                                   size_t n,
                                   ihipStream_t* stream)
                                   {
      if (n <= 0)
        return;
      thrust::counting_iterator<T> it(0);
      NonEmptyRow<T> temp_func(roff);
      T* d_out_num = (T*) tmp_get(sizeof(*n_nonempty), stream);

      cubIf(it, p_nonempty, d_out_num, n, temp_func, stream);
      hipMemcpy(n_nonempty, d_out_num, sizeof(*n_nonempty), hipMemcpyDeviceToHost);
      cudaCheckError();
      tmp_release(d_out_num, stream);
      cudaCheckError();
    }

    template<typename T>
    uint64_t reduce(uint64_t *v_d, T n, ihipStream_t* stream) {

      uint64_t n_h;
      uint64_t *n_d = (uint64_t *) tmp_get(sizeof(*n_d), stream);

      cubSum(v_d, n_d, n, stream);
      cudaCheckError();
      hipMemcpy(&n_h, n_d, sizeof(*n_d), hipMemcpyDeviceToHost);
      cudaCheckError();
      tmp_release(n_d, stream);

      return n_h;
    }

// instantiate for int
    template void tricnt_thr<int>(int nblock,
                                  spmat_t<int> *m,
                                  uint64_t *ocnt_d,
                                  ihipStream_t* stream);

    template uint64_t reduce<int>(uint64_t *v_d, int n, ihipStream_t* stream);
    template void create_nondangling_vector<int>(const int *roff,
                                                 int *p_nonempty,
                                                 int *n_nonempty,
                                                 size_t n,
                                                 ihipStream_t* stream);

  } // end namespace triangle counting

} // end namespace nvgraph
```

2. triangle_counting.cpp

```cpp
namespace nvgraph
{

namespace triangles_counting
{

template <typename IndexType>
TrianglesCount<IndexType>::TrianglesCount(const CsrGraph <IndexType>& graph, hipStream_t stream, int device_id)
{
    m_stream = stream;
    m_done = true;
    if (device_id == -1)
        hipGetDevice(&m_dev_id);
    else
        m_dev_id = device_id;

    hipGetDeviceProperties(&m_dev_props, m_dev_id);
    cudaCheckError();
    hipSetDevice(m_dev_id);
    cudaCheckError();

    // fill spmat struct;
    m_mat.nnz = graph.get_num_edges();
    m_mat.N = graph.get_num_vertices();
    m_mat.roff_d = graph.get_raw_row_offsets();
    m_mat.cols_d = graph.get_raw_column_indices();

    m_seq.allocate(m_mat.N, stream);
    create_nondangling_vector(m_mat.roff_d, m_seq.raw(), &(m_mat.nrows), m_mat.N, (ihipStream_t*)m_stream); 
    m_mat.rows_d = m_seq.raw();
}

template <typename IndexType>
TrianglesCount<IndexType>::~TrianglesCount()
{
    hipSetDevice(m_dev_id);
}

template <typename IndexType>
void TrianglesCount<IndexType>::tcount_thr()
{
//    printf("TrianglesCount: %s\n", __func__); fflush(stdout);

    int maxblocks = m_dev_props.multiProcessorCount * m_dev_props.maxThreadsPerMultiProcessor / THREADS;

    int nblock = MIN(maxblocks, DIV_UP(m_mat.nrows,THREADS));

    Vector<uint64_t> ocnt_d(nblock);

    hipMemset(ocnt_d.raw(), 0, ocnt_d.bytes());
    cudaCheckError();

    tricnt_thr(nblock, &m_mat, ocnt_d.raw(), (ihipStream_t*)m_stream);
    m_triangles_number = reduce(ocnt_d.raw(), nblock, (ihipStream_t*)m_stream);
}

template <typename IndexType>
NVGRAPH_ERROR TrianglesCount<IndexType>::count(TrianglesCountAlgo algo)
{
//  std::cout << "Starting TrianglesCount::count, Algo=" << algo << "\n";
    switch(algo)
    {
        case TCOUNT_THR:
            tcount_thr();
            break;
        default:
            FatalError("Bad algorithm specified for triangles counting", NVGRAPH_ERR_BAD_PARAMETERS);
    }
    m_event.record();
    return NVGRAPH_OK;
}

template class TrianglesCount<int>;

} // end namespace triangle counting

} // end namespace nvgraph
```

### api call

```cpp
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
```
