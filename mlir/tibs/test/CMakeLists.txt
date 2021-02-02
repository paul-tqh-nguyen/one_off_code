configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(PAULLANG_TEST_DEPENDS
        FileCheck count not
        paullang-opt
        paullang-translate
        )

add_lit_testsuite(check-paullang "Running the paullang regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${PAULLANG_TEST_DEPENDS}
        )
set_target_properties(check-paullang PROPERTIES FOLDER "Tests")

add_lit_testsuites(PAULLANG ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${PAULLANG_TEST_DEPENDS})
