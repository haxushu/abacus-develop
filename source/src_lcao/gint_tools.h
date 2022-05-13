//=========================================================
//REFACTOR : Peize Lin, 2021.06.28
//=========================================================
#ifndef GINT_TOOLS_H
#define GINT_TOOLS_H

#include <cstdlib>

namespace Gint_Tools
{
	template<typename T>
	class Array_Pool
	{
	public:
		T** ptr_2D;
		T* ptr_1D;
		Array_Pool(const int nr, const int nc);
		Array_Pool(Array_Pool<T> &&array);
		~Array_Pool();
		Array_Pool(const Array_Pool<T> &array) = delete;
		Array_Pool(Array_Pool<T> &array) = delete;
	};
	
	// vindex[pw.bxyz]
	int* get_vindex(
		const int ncyz,
		const int ibx,
		const int jby,
		const int kbz);

	// extract the local potentials.
	// vldr3[GlobalC::pw.bxyz]
    double* get_vldr3(
		const double* const vlocal,
		const int ncyz,
		const int ibx,
		const int jby,
		const int kbz,
		const double vfactor);

	//------------------------------------------------------
	// na_grid : #. atoms for this group of grids
	// block_iw : size na_grid, index of the first orbital on this atom
	// block_size : size na_grid, number of orbitals on this atom
	// block_index : size na_grid+1, start from 0, accumulates block_size
	//------------------------------------------------------
	void get_block_info(
		const int na_grid,
		const int grid_index,
		int * &block_iw,
		int * &block_index,
		int * &block_size
	);

	void get_block_info(
		const int na_grid,
		const int grid_index,
		int * &block_iw,
		int * &block_index,
		int * &block_size,
		int * &at,
		int * &uc
	);

	// whether the atom-grid distance is larger than cutoff
	// cal_flag[pw.bxyz][na_grid]
	bool** get_cal_flag(
		const int na_grid, 		// number of atoms on this grid 
		const int grid_index);		

	// psir_ylm[pw.bxyz][LD_pool]
	void cal_psir_ylm(
		const int na_grid, // number of atoms on this grid 
		const int grid_index, // 1d index of FFT index (i,j,k) 
		const double delta_r, // delta_r of the uniform FFT grid
		const int*const block_index,  // count total number of atomis orbitals
		const int*const block_size, 
		const bool*const*const cal_flag,
		double*const*const psir_ylm); // whether the atom-grid distance is larger than cutoff

	void cal_dpsir_ylm(
		const int na_grid, 					// number of atoms on this grid 
		const int grid_index, 				// 1d index of FFT index (i,j,k) 
		const double delta_r, 				// delta_r of the uniform FFT grid
		const int*const block_index,  		// block_index[na_grid+1], count total number of atomis orbitals
		const int*const block_size, 		// block_size[na_grid],	number of columns of a band
		const bool*const*const cal_flag,    // cal_flag[GlobalC::pw.bxyz][na_grid],	whether the atom-grid distance is larger than cutoff
		double*const*const psir_ylm,
		double*const*const dpsir_ylm_x,
		double*const*const dpsir_ylm_y,
		double*const*const dpsir_ylm_z);

	void cal_dpsirr_ylm(
		const int na_grid, 					// number of atoms on this grid 
		const int grid_index, 				// 1d index of FFT index (i,j,k) 
		const int*const block_index,  		// block_index[na_grid+1], count total number of atomis orbitals
		const int*const block_size, 		// block_size[na_grid],	number of columns of a band
		const bool*const*const cal_flag,    // cal_flag[GlobalC::pw.bxyz][na_grid],	whether the atom-grid distance is larger than cutoff
		double*const*const dpsir_ylm_x,
		double*const*const dpsir_ylm_y,
		double*const*const dpsir_ylm_z,
		double*const*const dpsir_ylm_xx,
		double*const*const dpsir_ylm_xy,
		double*const*const dpsir_ylm_xz,
		double*const*const dpsir_ylm_yy,
		double*const*const dpsir_ylm_yz,
		double*const*const dpsir_ylm_zz);

	Gint_Tools::Array_Pool<double> get_psir_vlbr3(
		const int na_grid,  					    // how many atoms on this (i,j,k) grid
		const int LD_pool,
		const int*const block_index,		    	// block_index[na_grid+1], count total number of atomis orbitals
		const bool*const*const cal_flag,	    	// cal_flag[GlobalC::pw.bxyz][na_grid],	whether the atom-grid distance is larger than cutoff
		const double*const vldr3,			    	// vldr3[GlobalC::pw.bxyz]
		const double*const*const psir_ylm);		    // psir_ylm[GlobalC::pw.bxyz][LD_pool]

	Gint_Tools::Array_Pool<double> get_psir_vlbr3_DM(
		const int na_grid,  					    // how many atoms on this (i,j,k) grid
		const int LD_pool,
		const int*const block_iw,				    // block_iw[na_grid],	index of wave functions for each block
		const int*const block_size, 			    // block_size[na_grid],	number of columns of a band
		const int*const block_index,		    	// block_index[na_grid+1], count total number of atomis orbitals
		const bool*const*const cal_flag,	    	// cal_flag[GlobalC::pw.bxyz][na_grid],	whether the atom-grid distance is larger than cutoff
		const double*const*const psir_vlbr3,	    // psir_vlbr3[GlobalC::pw.bxyz][LD_pool]
		const double*const*const DM);
}


namespace Gint_Tools
{
	template<typename T>
	Array_Pool<T>::Array_Pool(const int nr, const int nc)	// Attention: uninitialized
	{
		ptr_1D = (T*)malloc(nr*nc*sizeof(T));
		ptr_2D = (T**)malloc(nr*sizeof(T*));
		for (int ir=0; ir<nr; ++ir)
			ptr_2D[ir] = &ptr_1D[ir*nc];
	}

	template<typename T>
	Array_Pool<T>::Array_Pool(Array_Pool<T> &&array)
	{
		ptr_1D = array.ptr_1D;
		ptr_2D = array.ptr_2D;
		free(array.ptr_2D);		array.ptr_2D=nullptr;
		free(array.ptr_1D);		array.ptr_1D=nullptr;
	}

	template<typename T>
	Array_Pool<T>::~Array_Pool()
	{
		free(ptr_2D);
		free(ptr_1D);
	}
}

#endif