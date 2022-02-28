#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <complex>

extern "C"
int cusolver_DnDsygvd(int N, int M, double *A, double *B, double *W, double *V);

extern "C"
int cusolver_DnZhegvd(int N, int M, std::complex<double>  *A, std::complex<double>  *B, double *W, std::complex<double>  *V);
