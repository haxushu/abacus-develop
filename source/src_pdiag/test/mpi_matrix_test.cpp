#include "mini_pdiag.h"
#include "gtest/gtest.h"
#include <iostream>
#include <complex>

/************************************************
 *  unit test of class Pdiag_Double (Simplified version: Mini_Pdiag)
 ***********************************************/

/**
 * - Tested Functions:
 *   - Double_Whole
 *     - use rootDivide_double to divide a double matrix as a 2D block-cyclic layout
 * 	   - use rootGather_double to gather 2D block-cyclicly distributed double matrice to a whole one
 *     - functions: rootDivide_double and rootGather_double
 *   - Complex_Whole
 *     - use rootDivide_complex to divide a complex matrix as a 2D block-cyclic layout
 * 	   - use rootGather_complex to gather 2D block-cyclicly distributed complex matrice to a whole one
 *     - functions: rootDivide_complex and rootGather_complex
 * 
 */

const double eps = 1e-14;

TEST(MpiMatrixTest, Double_Whole)
{
	
	int dim = 1000;

	int myid;
	double *a, *b, *c;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);	
	if (myid == 0){
		a = new double[dim*dim];   // used to divide
		c = new double[dim*dim];   // used to gather

		int index = 0;
		for(int j = 0; j < dim; j++)
		{
			for(int i = 0; i< dim; i++)
			{
				a[j*dim+i] = 1.0*index++;
			}
		}
	}

	Mini_Pdiag mini_pdiag;

	mini_pdiag.init(dim);

	b = new double[mini_pdiag.nloc];
	mini_pdiag.rootDivide_double(a, b);
	mini_pdiag.rootGather_double(b, c);

	if (myid == 0){
		for(int j = 0; j < dim; j++)
		{
			for(int i = 0; i < dim; i++)
			{
				EXPECT_NEAR(a[j*dim+i], c[j*dim+i], eps);
			}
		}
	}
	delete[] b;
	if (myid == 0) {
		delete[] a;
		delete[] c;
	}
}

TEST(MpiMatrixTest, Complex_Whole)
{
	
	int dim = 1000;

	int myid;
	std::complex<double> *a, *b, *c;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);	
	if (myid == 0){
		a = new std::complex<double>[dim*dim];   // used to divide
		c = new std::complex<double>[dim*dim];   // used to gather

		int index = 0;
		for(int j = 0; j < dim; j++)
		{
			for(int i = 0; i< dim; i++)
			{
				a[j*dim+i] = {1.0*index, 1.0*index};
				index++;
			}
		}
	}

	Mini_Pdiag mini_pdiag;

	mini_pdiag.init(dim);

	b = new std::complex<double>[mini_pdiag.nloc];
	mini_pdiag.rootDivide_complex(a, b);
	mini_pdiag.rootGather_complex(b, c);

	if (myid == 0){
		for(int j = 0; j < dim; j++)
		{
			for(int i = 0; i < dim; i++)
			{
				EXPECT_NEAR(a[j*dim+i].real(), c[j*dim+i].real(), eps);
				EXPECT_NEAR(a[j*dim+i].imag(), c[j*dim+i].imag(), eps);
			}
		}
	}
	delete[] b;
	if (myid == 0) {
		delete[] a;
		delete[] c;
	}
}


int main(int argc, char **argv)
{

	MPI_Init(&argc,&argv);

	testing::InitGoogleTest(&argc,argv);
	int result = RUN_ALL_TESTS();

	MPI_Finalize();

	return result;
}

