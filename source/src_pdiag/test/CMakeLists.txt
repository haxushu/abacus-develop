IF (USE_CUSOLVER_LCAO)
AddTest(
  TARGET pdiag_cusolver
  LIBS -lcusolver
  SOURCES cusolver_test.cpp ../diag_cusolver.cu
)
endif()

AddTest(
  TARGET pdiag_mpi_matrix
  LIBS MPI::MPI_CXX
  SOURCES mpi_matrix_test.cpp mini_pdiag.cpp
)

