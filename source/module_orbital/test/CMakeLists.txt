remove_definitions(-D__MPI)

list(APPEND depend_files
 ../../module_base/math_integral.cpp
  ../../module_base/math_sphbes.cpp
  ../../module_base/math_polyint.cpp
  ../../module_base/math_ylmreal.cpp
  ../../module_base/ylm.cpp
  ../../module_base/memory.cpp
  ../../module_base/complexarray.cpp
  ../../module_base/complexmatrix.cpp
  ../../module_base/matrix.cpp
  ../../module_base/realarray.cpp
  ../../module_base/intarray.cpp
  ../../module_base/sph_bessel.cpp
  ../../module_base/sph_bessel_recursive-d1.cpp
  ../../module_base/sph_bessel_recursive-d2.cpp
  ../../module_base/tool_title.cpp
  ../../module_base/tool_quit.cpp
  ../../module_base/tool_check.cpp
  ../../module_base/timer.cpp
  ../../module_base/mathzone_add1.cpp
  ../../module_base/global_variable.cpp
  ../../module_base/global_function.cpp
  ../../module_base/global_file.cpp
  ../ORB_control.cpp
  ../ORB_read.cpp
  ../ORB_atomic.cpp
  ../ORB_atomic_lm.cpp
  ../ORB_nonlocal.cpp
  ../ORB_nonlocal_lm.cpp
  ../ORB_gaunt_table.cpp
  ../ORB_table_beta.cpp
  ../ORB_table_phi.cpp
  ../ORB_table_alpha.cpp
  ../ORB_gen_tables.cpp
  ../../src_lcao/center2_orb-orb11.cpp
  )
AddTest(
  TARGET orbital_equal_test
  LIBS ${math_libs}
  SOURCES 1_snap_equal_test.cpp ORB_unittest.cpp
  ${depend_files}
)
install(DIRECTORY GaAs DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../../../tests)
install(DIRECTORY GaAs DESTINATION ${CMAKE_CURRENT_BINARY_DIR})