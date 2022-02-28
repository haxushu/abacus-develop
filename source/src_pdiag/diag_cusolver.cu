#include "diag_cusolver.cuh"
#include "nvToolsExt.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>

void printMatrix(int m, int n, const double *A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){        // row dominant  
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = ", name, row+1, col+1);
            std::cout << Areg << std::endl;
             
        }
    } 
}

int cusolver_DnDsygvd(int N, int M, double *A, double *B, double *W, double *V)
{
    printf("enter begin: N=  %d M = %d \n", N, M);
    cudaEvent_t start_all, stop_all;
    cudaEventCreate(&start_all);
    cudaEventCreate(&stop_all);
    cudaEventRecord(start_all, 0);

    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;

    
    int m = N;
    int lda = m;

    double *d_A = NULL;
    double *d_B = NULL;

    double *d_W = NULL;
    int *devInfo = NULL;
    double *d_work = NULL;
    int  lwork = 0;
    int info_gpu = 0;
    // printf("A = (matlab base-1)\n");
    // printMatrix(m, m, A, lda, "A");
    // printf("=====\n");
    // printf("B = (matlab base-1)\n");
    // printMatrix(m, m, B, lda, "B");
    // printf("=====\n");

    // step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    // step 2: copy A and B to device
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
    cudaStat2 = cudaMalloc ((void**)&d_B, sizeof(double) * lda * m);
    cudaStat3 = cudaMalloc ((void**)&d_W, sizeof(double) * m);
    cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    // step 3: query working space of sygvd
    cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    //The helper functions below can calculate the sizes needed for pre-allocated buffer.
    //The S and D data types are real valued single and double precision, respectively.
    // The C and Z data types are complex valued single and double precision, respectively.
    cusolver_status = cusolverDnDsygvd_bufferSize(        
        cusolverH,
        itype,
        jobz,
        uplo,
        m,
        d_A,
        lda,
        d_B,
        lda,
        d_W,
        &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

    // step 4: compute spectrum of (A,B)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cusolver_status = cusolverDnDsygvd(
        cusolverH,
        itype,
        jobz,
        uplo,
        m,
        d_A,
        lda,
        d_B,
        lda,
        d_W,
        d_work,
        lwork,
        devInfo);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);     
    printf("time=%f ms\n",elapsedTime);

    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    // printf("after sygvd: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);
    // printf("eigenvalue = (matlab base-1), ascending order\n");
    // for(int i = 0 ; i < min(N,10) ; i++){
    //     printf("W[%d] = %E\n", i+1, W[i]);
    // }
    // for(int i = max(0, K-10) ; i < K ; i++){
    //     printf("W[%d] = %E\n", i+1, W[i]);
    // }
    // printf("V = (matlab base-1)\n");
    // printMatrix(m, m, V, lda, "V");
    // printf("=====\n");
    // step 4: check eigenvalues
    // double lambda_sup = 0;
    // for(int i = 0 ; i < m ; i++){
    //     double error = fabs( lambda[i] - W[i]);
    // }   lambda_sup = (lambda_sup > error)? lambda_sup : error;
    // printf("|lambda - W| = %E\n", lambda_sup);

    // free resources
    if (d_A    ) cudaFree(d_A);
    if (d_B    ) cudaFree(d_B);
    if (d_W    ) cudaFree(d_W);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    if (cusolverH) cusolverDnDestroy(cusolverH);

    cudaEventRecord(stop_all, 0);
    cudaEventSynchronize(stop_all);
    float elapsedTime_all;
    cudaEventElapsedTime(&elapsedTime_all, start_all, stop_all);
    cudaEventDestroy(start_all);
    cudaEventDestroy(stop_all);     
    printf("all_time=%f ms\n",elapsedTime_all);

    cudaDeviceReset();
    return 0; 
}



int cusolver_DnZhegvd(int N, int M, std::complex<double>  *A, std::complex<double>  *B, double *W, std::complex<double>  *V)
{
    printf("enter begin: N=  %d M = %d \n", N, M);
    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;

    
    int m = N;
    int lda = m;

    cuDoubleComplex *d_A = NULL;
    cuDoubleComplex *d_B = NULL;

    double *d_W = NULL;
    int *devInfo = NULL;
    cuDoubleComplex *d_work = NULL;
    int  lwork = 0;
    int info_gpu = 0;
    // printf("A = (matlab base-1)\n");
    // printMatrix(m, m, A, lda, "A");
    // printf("=====\n");
    // printf("B = (matlab base-1)\n");
    // printMatrix(m, m, B, lda, "B");
    // printf("=====\n");

    // step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    // step 2: copy A and B to device
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(cuDoubleComplex) * lda * m);
    cudaStat2 = cudaMalloc ((void**)&d_B, sizeof(cuDoubleComplex) * lda * m);
    cudaStat3 = cudaMalloc ((void**)&d_W, sizeof(double) * m);
    cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    cudaStat1 = cudaMemcpy(d_A, A, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    // step 3: query working space of sygvd
    cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    //The helper functions below can calculate the sizes needed for pre-allocated buffer.
    //The S and D data types are real valued single and double precision, respectively.
    // The C and Z data types are complex valued single and double precision, respectively.
    cusolver_status = cusolverDnZhegvd_bufferSize(        
        cusolverH,
        itype,
        jobz,
        uplo,
        m,
        d_A,
        lda,
        d_B,
        lda,
        d_W,
        &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex)*lwork);
    assert(cudaSuccess == cudaStat1);

    // step 4: compute spectrum of (A,B)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cusolver_status = cusolverDnZhegvd(
        cusolverH,
        itype,
        jobz,
        uplo,
        m,
        d_A,
        lda,
        d_B,
        lda,
        d_W,
        d_work,
        lwork,
        devInfo);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("time=%f\n",elapsedTime);

    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(V, d_A, sizeof(cuDoubleComplex)*lda*m, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    // printf("after sygvd: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);
    // printf("eigenvalue = (matlab base-1), ascending order\n");
    // for(int i = 0 ; i < min(N,10) ; i++){
    //     printf("W[%d] = %E\n", i+1, W[i]);
    // }
    // for(int i = max(0, N-10) ; i < N ; i++){
    //     printf("W[%d] = %E\n", i+1, W[i]);
    // }
    // printf("V = (matlab base-1)\n");
    // printMatrix(m, m, V, lda, "V");
    // printf("=====\n");
    // step 4: check eigenvalues
    // double lambda_sup = 0;
    // for(int i = 0 ; i < m ; i++){
    //     double error = fabs( lambda[i] - W[i]);
    // }   lambda_sup = (lambda_sup > error)? lambda_sup : error;
    // printf("|lambda - W| = %E\n", lambda_sup);

    // free resources
    if (d_A    ) cudaFree(d_A);
    if (d_B    ) cudaFree(d_B);
    if (d_W    ) cudaFree(d_W);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    if (cusolverH) cusolverDnDestroy(cusolverH);
    cudaDeviceReset();
    return 0; 
}

