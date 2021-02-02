configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TIBS_TEST_DEPENDS
        FileCheck count not
        tibs-opt
        tibs-translate
        )

add_lit_testsuite(check-tibs "Running the tibs regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${TIBS_TEST_DEPENDS}
        )
set_target_properties(check-tibs PROPERTIES FOLDER "Tests")

add_lit_testsuites(TIBS ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TIBS_TEST_DEPENDS})
