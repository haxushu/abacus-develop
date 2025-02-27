remove_definitions(-D__MPI -D__LCAO )
add_definitions(-D__CMD -D__NORMAL)

list(APPEND depend_files 
  ../MD_func.cpp
  ../cmd_neighbor.cpp
  ../LJ_potential.cpp
  ../DP_potential.cpp
  ../../input.cpp
  ../../module_cell/unitcell_pseudo.cpp
  ../../module_cell/unitcell.cpp
  ../../module_cell/atom_spec.cpp
  ../../module_cell/atom_pseudo.cpp
  ../../module_cell/read_atoms.cpp
  ../../module_cell/read_cell_pseudopots.cpp
  ../../module_cell/pseudo_nc.cpp
  ../../module_cell/read_pp.cpp
  ../../module_cell/read_pp_upf100.cpp
  ../../module_cell/read_pp_upf201.cpp
  ../../module_cell/read_pp_vwr.cpp
  ../../module_cell/read_pp_blps.cpp
  ../../module_base/matrix3.cpp
  ../../module_base/matrix.cpp
  ../../module_base/timer.cpp
  ../../module_base/memory.cpp
  ../../module_base/global_variable.cpp
  ../../module_base/global_function.cpp
  ../../module_base/global_file.cpp
  ../../module_base/tool_title.cpp
  ../../module_base/tool_quit.cpp
  ../../module_base/intarray.cpp
  ../../module_base/realarray.cpp
  ../../module_base/complexarray.cpp
  ../../module_neighbor/sltk_adjacent_set.cpp
  ../../module_neighbor/sltk_atom_arrange.cpp
  ../../module_neighbor/sltk_atom.cpp
  ../../module_neighbor/sltk_atom_input.cpp
  ../../module_neighbor/sltk_grid.cpp
  ../../module_neighbor/sltk_grid_driver.cpp
  ../../src_io/output.cpp
)

AddTest(
  TARGET md_LJ_pot
  LIBS ${math_libs}
  SOURCES LJ_pot_test.cpp  
  ${depend_files}
)

AddTest(
  TARGET md_func
  LIBS ${math_libs}
  SOURCES MD_func_test.cpp  
  ${depend_files}
)

AddTest(
  TARGET md_fire
  LIBS ${math_libs}
  SOURCES FIRE_test.cpp
  ../verlet.cpp
  ../FIRE.cpp
  ${depend_files}
)

AddTest(
  TARGET md_nve
  LIBS ${math_libs}
  SOURCES NVE_test.cpp
  ../verlet.cpp
  ../NVE.cpp
  ${depend_files}
)

AddTest(
  TARGET md_nvt_ads
  LIBS ${math_libs}
  SOURCES NVT_ADS_test.cpp
  ../verlet.cpp
  ../NVT_ADS.cpp
  ${depend_files}
)

AddTest(
  TARGET md_nvt_nhc
  LIBS ${math_libs}
  SOURCES NVT_NHC_test.cpp
  ../verlet.cpp
  ../NVT_NHC.cpp
  ${depend_files}
)

AddTest(
  TARGET md_msst
  LIBS ${math_libs}
  SOURCES MSST_test.cpp
  ../verlet.cpp
  ../MSST.cpp
  ${depend_files}
)

AddTest(
  TARGET md_lgv
  LIBS ${math_libs}
  SOURCES Langevin_test.cpp
  ../verlet.cpp
  ../Langevin.cpp
  ${depend_files}
)