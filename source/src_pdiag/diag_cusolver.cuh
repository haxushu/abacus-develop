#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <complex>

#include "nvToolsExt.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>


class Diag_cuSolver_gvd{

    cusolverDnHandle_t cusolverH;
    cusolverStatus_t cusolver_status;
    cudaError_t cudaStat1;
    cudaError_t cudaStat2;
    cudaError_t cudaStat3;
    cudaError_t cudaStat4;

    cusolverEigType_t itype; // A*x = (lambda)*B*x
    cusolverEigMode_t jobz; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo;

    int m;
    int lda;

    double *d_A;
    double *d_B;
    double *d_work;
    
    cuDoubleComplex *d_A2;
    cuDoubleComplex *d_B2;
    cuDoubleComplex *d_work2;

    double *d_W ;
    int *devInfo;

    int  lwork;
    int info_gpu;

public:

    int init_flag;

    template<typename T>
    void printMatrix(int m, int n, T *A, int lda, const char* name);

    void init_double(int N);
    void init_complex(int N);

    void copy_double(int N, int M, double *A, double *B);
    void copy_complex(int N, int M, std::complex<double> *A, std::complex<double> *B);

    void buffer_double();
    void buffer_complex();

    void compute_double();
    void compute_complex();


    void recopy_double(double *W, double *V);
    void recopy_complex(double *W, std::complex<double> *V);


    void finalize();
        
    int Dngvd_double(int N, int M, double *A, double *B, double *W, double *V);
    int Dngvd_complex(int N, int M, std::complex<double> *A, std::complex<double> *B, double *W, std::complex<double> *V);

    Diag_cuSolver_gvd();
    ~Diag_cuSolver_gvd();
};



// void Pdiag_Double::cuGather_double(const long maxnloc, double *mat_loc, double *mat_glb){
//     int myid;
//     MPI_Comm_rank(comm_2D, &myid);
//     int naroc[2]; 
//     double *work=new double[maxnloc]; // work/buffer matrix

//     for(int iprow=0; iprow<dim0; ++iprow)
//     {
//         for(int ipcol=0; ipcol<dim1; ++ipcol)
//         {
//             const int coord[2]={iprow, ipcol};
//             int src_rank;
//             MPI_Cart_rank(comm_2D, coord, &src_rank);
//             if(myid==src_rank)
//             {
//                 BlasConnector::copy(nloc, mat_loc, 1, work, 1);
//                 naroc[0]=nrow;
//                 naroc[1]=ncol;
//             }
//             info=MPI_Bcast(naroc, 2, MPI_INT, src_rank, comm_2D);
//             info=MPI_Bcast(work, maxnloc, MPI_DOUBLE, src_rank, comm_2D);


//             for(int j=0; j<naroc[1]; ++j)
//             {
//                 int igcol=globalIndex(j, nb, dim1, ipcol);
//                 for(int i=0; i<naroc[0]; ++i)
//                 {
//                     int igrow=globalIndex(i, nb, dim0, iprow);
//                     if (myid == 0) mat_glb[igcol*GlobalV::NLOCAL + igrow]=work[j*naroc[0]+i];
//                 }
//             }
//         }
//     }

//     delete[] work;   
// }

// void Pdiag_Double::cuGather_complex(const long maxnloc, const std::complex<double> *mat_loc, std::complex<double> *mat_glb){
//     int myid;
//     MPI_Comm_rank(comm_2D, &myid);
//     int naroc[2]; 
//     std::complex<double> *work=new std::complex<double>[maxnloc]; // work/buffer matrix
    
//     for(int iprow=0; iprow<dim0; ++iprow)
//     {
//         for(int ipcol=0; ipcol<dim1; ++ipcol)
//         {
//             const int coord[2]={iprow, ipcol};
//             int src_rank;
//             MPI_Cart_rank(comm_2D, coord, &src_rank);
//             if(myid==src_rank)
//             {
//                 BlasConnector::copy(nloc, mat_loc, 1, work, 1);
//                 naroc[0]=nrow;
//                 naroc[1]=ncol;
//             }
//             info=MPI_Bcast(naroc, 2, MPI_INT, src_rank, comm_2D);
//             info=MPI_Bcast(work, maxnloc, MPI_DOUBLE_COMPLEX, src_rank, comm_2D);


//             for(int j=0; j<naroc[1]; ++j)
//             {
//                 int igcol=globalIndex(j, nb, dim1, ipcol);
//                 for(int i=0; i<naroc[0]; ++i)
//                 {
//                     int igrow=globalIndex(i, nb, dim0, iprow);
//                     if (myid == 0) mat_glb[igcol*GlobalV::NLOCAL + igrow]=work[j*naroc[0]+i];
//                 }
//             }
//         }
//     }

//     delete[] work;   
// }

// void Pdiag_Double::cuDivide_double(const double *mat_glb, double *mat_loc){
//     int myid;
//     MPI_Comm_rank(comm_2D, &myid);
//     MPI_Bcast(mat_glb, GlobalV::NLOCAL*GlobalV::NLOCAL, MPI_DOUBLE, 0, comm_2D);
//     for(int iprow=0; iprow<dim0; ++iprow){
//         for(int ipcol=0; ipcol<dim1; ++ipcol){
//             const int coord[2]={iprow, ipcol};
//             int src_rank;
//             MPI_Cart_rank(comm_2D, coord, &src_rank);
//             if (src_rank != myid) continue;
//             for(int j=0; j<ncol; ++j)
//             {
//                 int igcol=globalIndex(j, nb, dim1, ipcol);
//                 for(int i=0; i<nrow; ++i)
//                 {
//                     int igrow=globalIndex(i, nb, dim0, iprow);
//                     mat_loc[j*nrow+i] = mat_glb[igcol*GlobalV::NLOCAL + igrow];
//                 }
//             }
//         }//loop ipcol
//     }//loop iprow
// }

// void Pdiag_Double::cuDivide_complex(const std::complex<double> *mat_glb, std::complex<double> *mat_loc){
//     int myid;
//     MPI_Comm_rank(comm_2D, &myid);
//     MPI_Bcast(mat_glb, GlobalV::NLOCAL*GlobalV::NLOCAL, MPI_DOUBLE_COMPLEX, 0, comm_2D);
//     for(int iprow=0; iprow<dim0; ++iprow){
//         for(int ipcol=0; ipcol<dim1; ++ipcol){
//             const int coord[2]={iprow, ipcol};
//             int src_rank;
//             MPI_Cart_rank(comm_2D, coord, &src_rank);
//             if (src_rank != myid) continue;
//             for(int j=0; j<ncol; ++j)
//             {
//                 int igcol=globalIndex(j, nb, dim1, ipcol);
//                 for(int i=0; i<nrow; ++i)
//                 {
//                     int igrow=globalIndex(i, nb, dim0, iprow);
//                     mat_loc[j*nrow+i] = mat_glb[igcol*GlobalV::NLOCAL + igrow];
//                 }
//             }
//         }//loop ipcol
//     }//loop iprow
// }

// template<typename T>
// void Diag_cuSolver_gvd::printMatrix(int m, int n, T *A, int lda, const char* name)
//     {
//         for(int row = 0 ; row < m ; row++){        // row dominant  
//             for(int col = 0 ; col < n ; col++){
//                 T Areg = A[row + col*lda];
//                 printf("%s(%d,%d) = ", name, row+1, col+1);
//                 std::cout << Areg << std::endl;
//             }
//         } 
//     }

// Diag_cuSolver_gvd::Diag_cuSolver_gvd(){
// 	// step 1: create cusolver/cublas handle
// 	cusolverH = NULL;
// 	cusolver_status = CUSOLVER_STATUS_SUCCESS;
// 	cudaStat1 = cudaSuccess;
// 	cudaStat2 = cudaSuccess;
// 	cudaStat3 = cudaSuccess;
// 	cudaStat4 = cudaSuccess;

// 	itype = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
// 	jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
// 	uplo = CUBLAS_FILL_MODE_LOWER;

// 	d_A = NULL;
// 	d_B = NULL;
// 	d_work = NULL;

// 	d_A2 = NULL;
// 	d_B2 = NULL;
// 	d_work2 = NULL;
	
// 	d_W = NULL;
// 	devInfo = NULL;

// 	lwork = 0;
// 	info_gpu = 0;
// }

// void Diag_cuSolver_gvd::init_double(int N){
// // step 2: Malloc A and B on device

// 	m = lda = N;
// 	istep = 0;
	
// 	cusolver_status = cusolverDnCreate(&cusolverH);
// 	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	
// 	cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
// 	cudaStat2 = cudaMalloc ((void**)&d_B, sizeof(double) * lda * m);
// 	cudaStat3 = cudaMalloc ((void**)&d_W, sizeof(double) * m);
// 	cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
// 	assert(cudaSuccess == cudaStat1);
// 	assert(cudaSuccess == cudaStat2);
// 	assert(cudaSuccess == cudaStat3);
// 	assert(cudaSuccess == cudaStat4);

// }


// void Diag_cuSolver_gvd::init_complex(int N){
// // step 2: Malloc A and B on device

// 	m = lda = N;
// 	istep = 0;
	
// 	cusolver_status = cusolverDnCreate(&cusolverH);
// 	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	
// 	cudaStat1 = cudaMalloc ((void**)&d_A2, sizeof(cuDoubleComplex) * lda * m);
// 	cudaStat2 = cudaMalloc ((void**)&d_B2, sizeof(cuDoubleComplex) * lda * m);
// 	cudaStat3 = cudaMalloc ((void**)&d_W, sizeof(double) * m);
// 	cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
// 	assert(cudaSuccess == cudaStat1);
// 	assert(cudaSuccess == cudaStat2);
// 	assert(cudaSuccess == cudaStat3);
// 	assert(cudaSuccess == cudaStat4);

// }


// void Diag_cuSolver_gvd::copy_double(int N, int M, double *A, double *B){
//         assert(N == M);
//         assert(M == m);
//         cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
//         assert(cudaSuccess == cudaStat1);
//         if (istep == 0) {
//             cudaStat2 = cudaMemcpy(d_B, B, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
//             assert(cudaSuccess == cudaStat2); 
//         }  
// }

// void Diag_cuSolver_gvd::copy_complex(int N, int M, std::complex<double> *A, std::complex<double> *B){
//         assert(N == M);
//         assert(M == m);
//         cudaStat1 = cudaMemcpy(d_A2, A, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyHostToDevice);
//         assert(cudaSuccess == cudaStat1);
//         if (istep == 0) {
//             cudaStat2 = cudaMemcpy(d_B2, B, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyHostToDevice);
//             assert(cudaSuccess == cudaStat2); 
//         }  
// }

// void Diag_cuSolver_gvd::buffer_double(){

//         // step 3: query working space of sygvd

//         //The helper functions below can calculate the sizes needed for pre-allocated buffer.
//         //The S and D data types are real valued single and double precision, respectively.
//         // The C and Z data types are complex valued single and double precision, respectively.
//         cusolver_status = cusolverDnDsygvd_bufferSize(        
//             cusolverH,
//             itype,
//             jobz,
//             uplo,
//             m,
//             d_A,
//             lda,
//             d_B,
//             lda,
//             d_W,
//             &lwork);

//         assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
//         cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
//         assert(cudaSuccess == cudaStat1);
// }

// void Diag_cuSolver_gvd::buffer_complex(){

//         // step 3: query working space of sygvd

//         //The helper functions below can calculate the sizes needed for pre-allocated buffer.
//         //The S and D data types are real valued single and double precision, respectively.
//         // The C and Z data types are complex valued single and double precision, respectively.

//         cusolver_status = cusolverDnZhegvd_bufferSize(        
//                 cusolverH,
//                 itype,
//                 jobz,
//                 uplo,
//                 m,
//                 d_A2,
//                 lda,
//                 d_B2,
//                 lda,
//                 d_W,
//                 &lwork);
                
//         assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
//         cudaStat1 = cudaMalloc((void**)&d_work2, sizeof(cuDoubleComplex)*lwork);
//         assert(cudaSuccess == cudaStat1);

// }

// void Diag_cuSolver_gvd::compute_double(){
//     // compute spectrum of (A,B)

//         cusolver_status = cusolverDnDsygvd(
//             cusolverH,
//             itype,
//             jobz,
//             uplo,
//             m,
//             d_A,
//             lda,
//             d_B,
//             lda,
//             d_W,
//             d_work,
//             lwork,
//             devInfo);

//         cudaStat1 = cudaDeviceSynchronize();
//         assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
//         assert(cudaSuccess == cudaStat1);
//     }


// void Diag_cuSolver_gvd::compute_complex(){
//     // compute spectrum of (A,B)

//         cusolver_status = cusolverDnZhegvd(
//             cusolverH,
//             itype,
//             jobz,
//             uplo,
//             m,
//             d_A2,
//             lda,
//             d_B2,
//             lda,
//             d_W,
//             d_work2,
//             lwork,
//             devInfo);

//         cudaStat1 = cudaDeviceSynchronize();
//         assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
//         assert(cudaSuccess == cudaStat1);
//     }


// void Diag_cuSolver_gvd::recopy_double(double *W, double *V){
//         cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
//         cudaStat2 = cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
//         cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
//         assert(cudaSuccess == cudaStat1);
//         assert(cudaSuccess == cudaStat2);
//         assert(cudaSuccess == cudaStat3);
//         // printf("after sygvd: info_gpu = %d\n", info_gpu);
//         assert(0 == info_gpu);
//         if (d_work ) cudaFree(d_work);
// }

// void Diag_cuSolver_gvd::recopy_complex(double *W, std::complex<double> *V){
//         cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
//         cudaStat2 = cudaMemcpy(V, d_A2, sizeof(std::complex<double>)*lda*m, cudaMemcpyDeviceToHost);
//         cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
//         assert(cudaSuccess == cudaStat1);
//         assert(cudaSuccess == cudaStat2);
//         assert(cudaSuccess == cudaStat3);
//         // printf("after sygvd: info_gpu = %d\n", info_gpu);
//         assert(0 == info_gpu);
//         if (d_work2 ) cudaFree(d_work);
// }


// void Diag_cuSolver_gvd::finalize(){
//         // free resources and destroy
//         if (d_A    ) cudaFree(d_A);
//         if (d_B    ) cudaFree(d_B);
//         if (d_A2    ) cudaFree(d_A);
//         if (d_B2    ) cudaFree(d_B);
//         if (d_W    ) cudaFree(d_W);
//         if (devInfo) cudaFree(devInfo);
//         if (cusolverH) cusolverDnDestroy(cusolverH);
//         cudaDeviceReset();
//     }
        
// Diag_cuSolver_gvd::~Diag_cuSolver_gvd(){
//     finalize();
// }
// int Diag_cuSolver_gvd::Dngvd_double(int N, int M, double *A, double *B, double *W, double *V){
//         // printf("A = (matlab base-1)\n");
//         // printMatrix(m, m, A, lda, "A");
//         // printf("=====\n");
//         // printf("B = (matlab base-1)\n");
//         // printMatrix(m, m, B, lda, "B");
//         // printf("=====\n");

//         copy_double(N, M, A, B);
//         buffer_double();
//         compute_double();
//         recopy_double(W, V);
// 		// istep++;

    
//         // printf("eigenvalue = (matlab base-1), ascending order\n");
//         // for(int i = 0 ; i < std::min(N,10) ; i++){
//         //     printf("W[%d] = %E\n", i+1, W[i]);
//         // }
//         // for(int i = max(0, K-10) ; i < K ; i++){
//         //     printf("W[%d] = %E\n", i+1, W[i]);
//         // }
//         // printf("V = (matlab base-1)\n");
//         // printMatrix(m, m, V, lda, "V");
//         // printf("=====\n");
//         // step 4: check eigenvalues
//         // double lambda_sup = 0;
//         // for(int i = 0 ; i < m ; i++){
//         //     double error = fabs( lambda[i] - W[i]);
//         // }   lambda_sup = (lambda_sup > error)? lambda_sup : error;
//         // printf("|lambda - W| = %E\n", lambda_sup);

//         return 0; 
//     }


// int Diag_cuSolver_gvd::Dngvd_complex(int N, int M, std::complex<double> *A, std::complex<double> *B, double *W, std::complex<double> *V){
//         // printf("A = (matlab base-1)\n");
//         // printMatrix(m, m, A, lda, "A");
//         // printf("=====\n");
//         // printf("B = (matlab base-1)\n");
//         // printMatrix(m, m, B, lda, "B");
//         // printf("=====\n");

//         copy_complex(N, M, A, B);
//         buffer_complex();
//         compute_complex();
//         recopy_complex(W, V);
// 		// istep++;

    
//         // printf("eigenvalue = (matlab base-1), ascending order\n");
//         // for(int i = 0 ; i < std::min(N,10) ; i++){
//         //     printf("W[%d] = %E\n", i+1, W[i]);
//         // }
//         // for(int i = max(0, K-10) ; i < K ; i++){
//         //     printf("W[%d] = %E\n", i+1, W[i]);
//         // }
//         // printf("V = (matlab base-1)\n");
//         // printMatrix(m, m, V, lda, "V");
//         // printf("=====\n");
//         // step 4: check eigenvalues
//         // double lambda_sup = 0;
//         // for(int i = 0 ; i < m ; i++){
//         //     double error = fabs( lambda[i] - W[i]);
//         // }   lambda_sup = (lambda_sup > error)? lambda_sup : error;
//         // printf("|lambda - W| = %E\n", lambda_sup);

//         return 0; 
//     }
