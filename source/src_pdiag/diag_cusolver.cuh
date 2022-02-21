#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <iostream>
#include <fstream>

#ifdef __CUSOLVER
// #include "cufft.h"
// #include "cublas_v2.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#endif 

extern "C"
int cusolver_DnDsygvd(int N, int M, double *A, double *B, double *W, double *V);