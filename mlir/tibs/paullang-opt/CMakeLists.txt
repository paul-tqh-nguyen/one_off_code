get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRPaulLang
        )
add_llvm_executable(paullang-opt paullang-opt.cpp)

llvm_update_compile_flags(paullang-opt)
target_link_libraries(paullang-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(paullang-opt)

#message(STATUS "=======================================")
#message(STATUS "paullang-opt")
#message(STATUS "=======================================")
#message(STATUS "examples/paul/paullang-opt/CMakeLists.txt")
#message(STATUS "dialect_libs " ${dialect_libs})
#message(STATUS "conversion_libs " ${conversion_libs})
#message(STATUS "LIBS " ${LIBS})
#message(STATUS "=======================================")

