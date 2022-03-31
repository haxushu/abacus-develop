#include <complex>
#include "mpi.h"
class Mini_Pdiag
{
	public:

    int NLOCAL;
	int nrow;
	int ncol;
	long nloc;

	MPI_Comm comm_2D;

	int nb;
	int dim0;
	int dim1;

	void init(int nlocal);	// used to simplify Pdiag_Double to Mini_Pdiag
    
	// Diag_Cusolver_gvd diag_cusolver_gvd;	// test at the other unit-test "cusolver_test"
	void rootGather_double(double *mat_loc, double *mat_glb);
	void rootGather_complex(std::complex<double> *mat_loc, std::complex<double> *mat_glb);
	void rootDivide_double(double *mat_glb, double *mat_loc);
	void rootDivide_complex(std::complex<double> *mat_glb, std::complex<double> *mat_loc);

};