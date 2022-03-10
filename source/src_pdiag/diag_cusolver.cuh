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



