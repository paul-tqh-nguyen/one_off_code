add_mlir_dialect_library(MLIRTibs
        TibsDialect.cpp
        TibsPasses.cpp
        TibsOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Tibs

        DEPENDS
        MLIRTibsOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)

message(STATUS "ADDITIONAL_HEADER_DIRS" ${ADDITIONAL_HEADER_DIRS})
message(STATUS "PROJECT_SOURCE_DIR" ${PROJECT_SOURCE_DIR})
message(STATUS "DEPENDS" ${DEPENDS})
message(STATUS "LINK_LIBS" ${LINK_LIBS})
