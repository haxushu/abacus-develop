#ifndef LOCAL_ORBITAL_WFC
#define LOCAL_ORBITAL_WFC

#include "grid_technique.h"
#include "../src_pw/tools.h"

class Local_Orbital_wfc
{
	public:
	Local_Orbital_wfc();
	~Local_Orbital_wfc();

	double*** WFC_GAMMA; // [NSPIN, NBANDS, NLOCAL]
	complex<double>*** WFC_GAMMA_B; // WFC_GAMMA for B field calculation.

	// used to generate density matrix: LOC.DM_R,
	// which is used to calculate the charge density. 
	// which is got after the diagonalization of 
	// complex Hamiltonian matrix.
	complex<double>*** WFC_K; // [NK, NBANDS, NLOCAL]	
	complex<double>* WFC_K_POOL; // [NK*NBANDS*NLOCAL]

	// augmented wave functions to 'c',
	// used to generate density matrix 
	// according to 2D data block.
	// mohan add 2010-09-26
	// daug means : dimension of augmented wave functions
	double*** WFC_GAMMA_aug; // [NSPIN, NBANDS, daug];
	complex<double>*** WFC_K_aug; // [NK, NBANDS, daug];
	int* trace_aug;
	
	// how many elements are missing. 
	int daug;

	void allocate_k(const Grid_Technique &gt);
	void aloc_gamma_wfc(const Grid_Technique &gt);
	void set_trace_aug(const Grid_Technique &gt);
	const bool get_allocate_aug_flag(void)const{return allocate_aug_flag;};

    //=========================================
    // Init Cij, make it satisfy 2 conditions:
    // (1) Unit
    // (2) Orthogonal <i|S|j>= \delta{ij}
    //=========================================
	void init_Cij(const bool change_c = 1);
	bool get_allocate_flag(void)const{return allocate_flag;}	
	
	private:

	bool wfck_flag; 
	bool complex_flag;
	bool allocate_flag;
	bool allocate_aug_flag;
	int check_orthogonal(double *psi, double *sc, const int &is);

};

#endif
