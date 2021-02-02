add_mlir_dialect_library(MLIRPaulLang
        PaulLangDialect.cpp
        PaulLangPasses.cpp
        PaulLangOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/PaulLang

        DEPENDS
        MLIRPaulLangOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)

message(STATUS "ADDITIONAL_HEADER_DIRS" ${ADDITIONAL_HEADER_DIRS})
message(STATUS "PROJECT_SOURCE_DIR" ${PROJECT_SOURCE_DIR})
message(STATUS "DEPENDS" ${DEPENDS})
message(STATUS "LINK_LIBS" ${LINK_LIBS})
