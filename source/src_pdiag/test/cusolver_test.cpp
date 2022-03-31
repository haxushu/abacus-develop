#include "../diag_cusolver.cuh"
#include "gtest/gtest.h"
#include <iostream>
#include <complex>
#include <cmath>

/************************************************
 *  unit test of class Diag_Cusolver_gvd
 ***********************************************/

/**
 * - Tested Functions:
 *   - Double
 *     - use Dngvd_double to solve the generalized eigenvalue problem for a Hermitian matrix H and a SPD matrix S
 *     - functions: init_double and Dngvd_double
 *   - Complex
 *     - use Dngvd_complex to solve the generalized eigenvalue problem for a Hermitian matrix H and a SPD matrix S
 *     - functions: init_complex and Dngvd_complex
 *
 */

const double eps = 1e-14;

TEST(CusolverTest, Double)
{
	int dim = 2;

	Diag_Cusolver_gvd diag_cusolver_gvd;
	diag_cusolver_gvd.init_double(dim);
	
	double* a = new double[dim*dim];	// Hermitian matrix
	double* b = new double[dim*dim];	// SPD matrix
	double* c = new double[dim];		// eigenvalues
	double* d = new double[dim*dim];    // eigenvectors
	double eig[dim];					// reference
	
	// construct the matrix A and B w.r.t the problem Ax = \lambda Bx
	for (int i = 0; i < dim*dim; i++) a[i] = 1;
	b[0] = b[3] = 1;
	// the reference solution 
	eig[0] = 0; eig[1] = 2;

	diag_cusolver_gvd.Dngvd_double(dim, dim, a, b, c, d);

	for(int i=0;i<dim;i++) EXPECT_NEAR(c[i], eig[i], eps);

	delete[] a;
	delete[] b;
	delete[] c;
	delete[] d;
}


TEST(CusolverTest, Complex_1)
{
	int dim = 2;

	Diag_Cusolver_gvd diag_cusolver_gvd;
	diag_cusolver_gvd.init_complex(dim);
	
	std::complex<double>* a = new std::complex<double>[dim*dim];	// Hermitian matrix
	std::complex<double>* b = new std::complex<double>[dim*dim];	// SPD matrix
	std::complex<double>* d = new std::complex<double>[dim*dim];	// eigenvectors
	double* c = new double[dim];	// eigenvalues
	double eig[dim];				// reference
	
	// construct the matrix A and B w.r.t the problem Ax = \lambda Bx
	for (int i = 0; i < dim*dim; i++) a[i] = {1, 0};
	b[0] = b[3] = {1,0};
	// the reference solution 
	eig[0] = 0; eig[1] = 2; 

	diag_cusolver_gvd.Dngvd_complex(dim, dim, a, b, c, d);

	for(int i=0;i<dim;i++) EXPECT_NEAR(c[i], eig[i], eps);

	delete[] a;
	delete[] b;
	delete[] c;
	delete[] d;
}

TEST(CusolverTest, Complex_2)
{
	int dim = 2;

	Diag_Cusolver_gvd diag_cusolver_gvd;
	diag_cusolver_gvd.init_complex(dim);
	
	std::complex<double>* a = new std::complex<double>[dim*dim];	// Hermitian matrix
	std::complex<double>* b = new std::complex<double>[dim*dim];	// SPD matrix
	std::complex<double>* d = new std::complex<double>[dim*dim];	// eigenvectors
	double* c = new double[dim];	// eigenvalues
	double eig[dim];				// reference
	
	// construct the matrix A and B w.r.t the problem Ax = \lambda Bx
	a[0] = a[3] = {1, 0};
	a[1] = {1, sqrt(3)};
	a[2] = {1, -sqrt(3)};

	b[0] = b[3] = {1,0};
	
	// the reference solution 
	eig[0] = -1; eig[1] = 3; 

	diag_cusolver_gvd.Dngvd_complex(dim, dim, a, b, c, d);

	for(int i=0;i<dim;i++) EXPECT_NEAR(c[i], eig[i], eps);

	delete[] a;
	delete[] b;
	delete[] c;
	delete[] d;
}
