if(ENABLE_DEEPKS)
    list(APPEND objects
        LCAO_deepks.cpp
        LCAO_deepks_fdelta.cpp
        LCAO_deepks_io.cpp
        LCAO_deepks_mpi.cpp
        LCAO_deepks_pdm.cpp
        LCAO_deepks_psialpha.cpp
        LCAO_deepks_torch.cpp
        LCAO_deepks_vdelta.cpp
        )

    add_library(
        deepks
        OBJECT
        ${objects}
    )
endif()

