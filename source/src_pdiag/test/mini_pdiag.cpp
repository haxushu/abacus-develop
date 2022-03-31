#include <complex>
#include <cstring>
#include "mpi.h"

#include "mini_pdiag.h"

void Mini_Pdiag::init(int nlocal){
    NLOCAL = nlocal;
    comm_2D = MPI_COMM_WORLD;
    nb = 1;
    dim0 = 1;
    MPI_Comm_size(comm_2D, &dim1);

    int myid;
    MPI_Comm_rank(comm_2D, &myid);

    nrow = nlocal;
	ncol = nlocal/dim1 + (myid < (nlocal%dim1) ? 1 : 0)  ;
	nloc = nrow * ncol;
}

void Mini_Pdiag::rootGather_double(double *mat_loc, double *mat_glb){
    
	int myid;
    MPI_Comm_rank(comm_2D, &myid);

	int maxncol;

	MPI_Allreduce(&ncol, &maxncol, 1, MPI_INT, MPI_MAX, comm_2D);

	int displs[dim1], rcounts[dim1];
	for (int j = 0; j < dim1; j++ ) 
	{
		rcounts[j] = nrow;
		displs[j] = j * nrow ;
	}

	for(int i = 0; i < maxncol - 1; i++){
		MPI_Gatherv(mat_loc + i*nrow, nrow, MPI_DOUBLE, mat_glb + i*dim1*nrow, rcounts, displs, MPI_DOUBLE, 0, comm_2D);   
	}

	if (NLOCAL % dim1) 
		for (int j = NLOCAL % dim1; j < dim1; j++ ) rcounts[j] = 0;

	MPI_Gatherv(mat_loc + (maxncol - 1)*nrow, rcounts[myid], MPI_DOUBLE, mat_glb + (maxncol - 1)*dim1*nrow, rcounts, displs, MPI_DOUBLE, 0, comm_2D);  

}



void Mini_Pdiag::rootGather_complex(std::complex<double> *mat_loc, std::complex<double> *mat_glb){

	int myid;
    MPI_Comm_rank(comm_2D, &myid);

	int maxncol;

	MPI_Allreduce(&ncol, &maxncol, 1, MPI_INT, MPI_MAX, comm_2D);

	int displs[dim1], rcounts[dim1];
	for (int j = 0; j < dim1; j++ ) 
	{
		rcounts[j] = nrow;
		displs[j] = j * nrow ;
	}

	for(int i = 0; i < maxncol - 1; i++){
		MPI_Gatherv(mat_loc + i*nrow, nrow, MPI_DOUBLE_COMPLEX, mat_glb + i*dim1*nrow, rcounts, displs, MPI_DOUBLE_COMPLEX, 0, comm_2D);   
	}

	if (NLOCAL % dim1) 
		for (int j = NLOCAL % dim1; j < dim1; j++ ) rcounts[j] = 0;

	MPI_Gatherv(mat_loc + (maxncol - 1)*nrow, rcounts[myid], MPI_DOUBLE_COMPLEX, mat_glb + (maxncol - 1)*dim1*nrow, rcounts, displs, MPI_DOUBLE_COMPLEX, 0, comm_2D);  

}

void Mini_Pdiag::rootDivide_double(double *mat_glb, double *mat_loc){
    int myid;
    MPI_Comm_rank(comm_2D, &myid);
    MPI_Status status;

	if (myid == 0){
		for (int i =0; i < NLOCAL; i++){
			if ((i % dim1) == 0) continue;
			MPI_Send(mat_glb + i*nrow, nrow, MPI_DOUBLE, i%dim1, i/dim1, comm_2D);
		}
		for (int i =0; i < NLOCAL; i+=dim1)
			memcpy(mat_loc + i/dim1*nrow, mat_glb + i*nrow, nrow*sizeof(double));
	} else {
		for (int i = 0; i < ncol; i++)
			MPI_Recv(mat_loc + i*nrow, nrow, MPI_DOUBLE, 0, i, comm_2D, &status);
	}

}

void Mini_Pdiag::rootDivide_complex(std::complex<double> *mat_glb, std::complex<double> *mat_loc){
    int myid;
    MPI_Comm_rank(comm_2D, &myid);
    MPI_Status status;

	if (myid == 0){
		for (int i =0; i < NLOCAL; i++){
			if ((i % dim1) == 0) continue;
			MPI_Send(mat_glb + i*nrow, nrow, MPI_DOUBLE_COMPLEX, i%dim1, i/dim1, comm_2D);
		}
		for (int i =0; i < NLOCAL; i+=dim1)
			memcpy(mat_loc + i/dim1*nrow, mat_glb + i*nrow, nrow*sizeof(std::complex<double>));
	} else {
		for (int i = 0; i < ncol; i++)
			MPI_Recv(mat_loc + i*nrow, nrow, MPI_DOUBLE_COMPLEX, 0, i, comm_2D, &status);
	}
}
