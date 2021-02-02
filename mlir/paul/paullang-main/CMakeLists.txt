
add_executable(paullang-main paullang-main.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS) # makes it so we include llvm-project/mlir/lib/Dialect
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS) # makes it so we include llvm-project/mlir/lib/Conversion
set(LIBS
        ${dialect_libs}
	${conversion_libs}
        )
llvm_update_compile_flags(paullang-main) # TODO figure out what this does
target_link_libraries(paullang-main PRIVATE ${LIBS})

#message(STATUS "=======================================")
#message(STATUS "paullang-main")
#message(STATUS "=======================================")
#message(STATUS "dialect_libs: ${dialect_libs}")
#message(STATUS "=======================================")
#message(STATUS "conversion_libs: ${conversion_libs}")
#message(STATUS "=======================================")

