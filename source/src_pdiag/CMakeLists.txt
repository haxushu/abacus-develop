add_subdirectory(MRRR)

if(USE_CUSOLVER)
    list(APPEND objects
        pdiag_double.cu
    )
else()
    list(APPEND objects
    pdiag_double.cpp
    )
endif()

add_library(
    pdiag
    OBJECT
    pdgseps.cpp
    pdiag_basic.cpp
    pdiag_common.cpp
    ${objects}
    pdst2g.cpp
    pdstebz.cpp
    pdsteiz.cpp
    pdsyg2st.cpp
    pdsytrd.cpp
    pdt2s.cpp
    pdtrsm.cpp
    pzgseps.cpp
    pzheg2st.cpp
    pzhetrd.cpp
    pzhtrsm.cpp
    pzst2g.cpp
    pzsteiz.cpp
    pzt2s.cpp
    diag_scalapack_gvx.cpp
)

