#include "diag_cusolver.cuh"
#include <assert.h>
Diag_cuSolver_gvd::Diag_cuSolver_gvd(){
	// step 1: create cusolver/cublas handle
	cusolverH = NULL;
	cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaStat1 = cudaSuccess;
	cudaStat2 = cudaSuccess;
	cudaStat3 = cudaSuccess;
	cudaStat4 = cudaSuccess;

	itype = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
	jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
	uplo = CUBLAS_FILL_MODE_LOWER;

	d_A = NULL;
	d_B = NULL;
	d_work = NULL;

	d_A2 = NULL;
	d_B2 = NULL;
	d_work2 = NULL;
	
	d_W = NULL;
	devInfo = NULL;

	lwork = 0;
	info_gpu = 0;
	init_flag = 0;
}

void Diag_cuSolver_gvd::init_double(int N){
// step 2: Malloc A and B on device

	m = lda = N;
	
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	
	cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
	cudaStat2 = cudaMalloc ((void**)&d_B, sizeof(double) * lda * m);
	cudaStat3 = cudaMalloc ((void**)&d_W, sizeof(double) * m);
	cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);

}


void Diag_cuSolver_gvd::init_complex(int N){
// step 2: Malloc A and B on device

	m = lda = N;
	
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	
	cudaStat1 = cudaMalloc ((void**)&d_A2, sizeof(cuDoubleComplex) * lda * m);
	cudaStat2 = cudaMalloc ((void**)&d_B2, sizeof(cuDoubleComplex) * lda * m);
	cudaStat3 = cudaMalloc ((void**)&d_W, sizeof(double) * m);
	cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);

}


void Diag_cuSolver_gvd::copy_double(int N, int M, double *A, double *B){
        assert(N == M);
        assert(M == m);
        cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
        assert(cudaSuccess == cudaStat1);
        cudaStat2 = cudaMemcpy(d_B, B, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
        assert(cudaSuccess == cudaStat2); 
}

void Diag_cuSolver_gvd::copy_complex(int N, int M, std::complex<double> *A, std::complex<double> *B){
        assert(N == M);
        assert(M == m);
        cudaStat1 = cudaMemcpy(d_A2, A, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyHostToDevice);
        assert(cudaSuccess == cudaStat1);
		cudaStat2 = cudaMemcpy(d_B2, B, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyHostToDevice);
		assert(cudaSuccess == cudaStat2); 
}

void Diag_cuSolver_gvd::buffer_double(){
    // step 3: query working space of sygvd

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
}

void Diag_cuSolver_gvd::buffer_complex(){
    // step 3: query working space of sygvd

    //The helper functions below can calculate the sizes needed for pre-allocated buffer.
    //The S and D data types are real valued single and double precision, respectively.
    // The C and Z data types are complex valued single and double precision, respectively.

    cusolver_status = cusolverDnZhegvd_bufferSize(        
            cusolverH,
            itype,
            jobz,
            uplo,
            m,
            d_A2,
            lda,
            d_B2,
            lda,
            d_W,
            &lwork);
            
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    cudaStat1 = cudaMalloc((void**)&d_work2, sizeof(cuDoubleComplex)*lwork);
    assert(cudaSuccess == cudaStat1);

}

void Diag_cuSolver_gvd::compute_double(){
    // compute spectrum of (A,B)

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

        cudaStat1 = cudaDeviceSynchronize();
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
        assert(cudaSuccess == cudaStat1);
    }


void Diag_cuSolver_gvd::compute_complex(){
    // compute spectrum of (A,B)

        cusolver_status = cusolverDnZhegvd(
            cusolverH,
            itype,
            jobz,
            uplo,
            m,
            d_A2,
            lda,
            d_B2,
            lda,
            d_W,
            d_work2,
            lwork,
            devInfo);

        cudaStat1 = cudaDeviceSynchronize();
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
        assert(cudaSuccess == cudaStat1);
    }


void Diag_cuSolver_gvd::recopy_double(double *W, double *V){
        cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
        cudaStat2 = cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
        cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        assert(cudaSuccess == cudaStat1);
        assert(cudaSuccess == cudaStat2);
        assert(cudaSuccess == cudaStat3);
        assert(0 == info_gpu);
        if (d_work ) cudaFree(d_work);
}

void Diag_cuSolver_gvd::recopy_complex(double *W, std::complex<double> *V){
        cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
        cudaStat2 = cudaMemcpy(V, d_A2, sizeof(std::complex<double>)*lda*m, cudaMemcpyDeviceToHost);
        cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        assert(cudaSuccess == cudaStat1);
        assert(cudaSuccess == cudaStat2);
        assert(cudaSuccess == cudaStat3);
        assert(0 == info_gpu);
        if (d_work2 ) cudaFree(d_work);
}


void Diag_cuSolver_gvd::finalize(){
        // free resources and destroy
        if (d_A    ) cudaFree(d_A);
        if (d_B    ) cudaFree(d_B);
        if (d_A2    ) cudaFree(d_A);
        if (d_B2    ) cudaFree(d_B);
        if (d_W    ) cudaFree(d_W);
        if (devInfo) cudaFree(devInfo);
        if (cusolverH) cusolverDnDestroy(cusolverH);
        cudaDeviceReset();
    }
        
Diag_cuSolver_gvd::~Diag_cuSolver_gvd(){
    finalize();
}
int Diag_cuSolver_gvd::Dngvd_double(int N, int M, double *A, double *B, double *W, double *V){

        copy_double(N, M, A, B);
        buffer_double();
        compute_double();
        recopy_double(W, V);

        return 0; 
}


int Diag_cuSolver_gvd::Dngvd_complex(int N, int M, std::complex<double> *A, std::complex<double> *B, double *W, std::complex<double> *V){

        copy_complex(N, M, A, B);
        buffer_complex();
        compute_complex();
        recopy_complex(W, V);

        return 0; 
}