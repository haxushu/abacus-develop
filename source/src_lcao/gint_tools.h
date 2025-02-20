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

	// index of wave functions for each block
	// block_iw[na_grid]
	int* get_block_iw(
		const int na_grid,  		// how many atoms on this (i,j,k) grid
		const int grid_index,		// 1d index of FFT index (i,j,k))
		const int max_size);
		
	// block_index[na_grid+1]
	int* get_block_index(
		const int na_grid,  		// how many atoms on this (i,j,k) grid
		const int grid_index);		// 1d index of FFT index (i,j,k)
		
	// band size: number of columns of a band
	// block_size[na_grid]
	int* get_block_size(
		const int na_grid,			// how many atoms on this (i,j,k) grid
		const int grid_index);		// 1d index of FFT index (i,j,k)

	// whether the atom-grid distance is larger than cutoff
	// cal_flag[pw.bxyz][na_grid]
	bool** get_cal_flag(
		const int na_grid, 		// number of atoms on this grid 
		const int grid_index);		

	// psir_ylm[pw.bxyz][LD_pool]
	Array_Pool<double> cal_psir_ylm(
		const int na_grid, // number of atoms on this grid 
		const int LD_pool,
		const int grid_index, // 1d index of FFT index (i,j,k) 
		const double delta_r, // delta_r of the uniform FFT grid
		const int*const block_index,  // count total number of atomis orbitals
		const int*const block_size, 
		const bool*const*const cal_flag); // whether the atom-grid distance is larger than cutoff		
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