#ifndef GLOBAL_FP_H
#define GLOBAL_FP_H

#include "../module_neighbor/sltk_grid_driver.h"
#include "grid_technique.h"
#include "../src_parallel/parallel_orbitals.h"
#include "local_orbital_wfc.h"
#include "local_orbital_charge.h"
#include "LCAO_matrix.h"
#include "LCAO_gen_fixedH.h"
#include "LCAO_hamilt.h" 
#include "../module_orbital/ORB_read.h"
#include "../module_orbital/ORB_gen_tables.h"
#include "../src_parallel/subgrid_oper.h"
#include "../src_ri/exx_lcao.h"

namespace GlobalC
{
extern Grid_Driver GridD;
extern Parallel_Orbitals ParaO;
extern Local_Orbital_Charge LOC;
extern Local_Orbital_wfc LOWF;
extern LCAO_Matrix LM;
extern LCAO_Hamilt UHM;
extern SubGrid_oper SGO; //mohan add 2012-01-12
extern Exx_Lcao exx_lcao; // Peize Lin add 2016-12-03
}
#endif
