get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        )
add_llvm_executable(tibs-opt tibs-opt.cpp)

llvm_update_compile_flags(tibs-opt)
target_link_libraries(tibs-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(tibs-opt)

#message(STATUS "=======================================")
#message(STATUS "tibs-opt")
#message(STATUS "=======================================")
#message(STATUS "tibs-opt/CMakeLists.txt")
#message(STATUS "dialect_libs " ${dialect_libs})
#message(STATUS "conversion_libs " ${conversion_libs})
#message(STATUS "LIBS " ${LIBS})
#message(STATUS "=======================================")

