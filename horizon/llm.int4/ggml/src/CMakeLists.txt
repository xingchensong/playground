message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

# ggml
set(TARGET ggml)

add_library(${TARGET}
  ggml.c
  ggml-alloc.c
  ggml-backend.c
  ggml-quants.c
  ggml-impl.h
  ggml-backend-impl.h
  ../include/ggml/ggml.h
  ../include/ggml/ggml-alloc.h
  ../include/ggml/ggml-backend.h
)

target_include_directories(${TARGET} PUBLIC
  .
  ../include
  ../include/ggml
)

target_link_libraries(
  ${TARGET} PUBLIC m ${GGML_EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT})

set(GGML_PUBLIC_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/ggml/ggml.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/ggml/ggml-alloc.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/ggml/ggml-backend.h)

set_target_properties(${TARGET} PROPERTIES
                      PUBLIC_HEADER "${GGML_PUBLIC_HEADERS}")

add_library(common STATIC common.cpp)
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(common-ggml STATIC common-ggml.cpp)
target_link_libraries(common-ggml PRIVATE ${TARGET})
target_include_directories(common-ggml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

