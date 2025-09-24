get_filename_component(CURRENT_FOLDER_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
set(LIB_NAME "light")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.9.2
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(${LIB_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/light.cpp")

file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
"${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)
target_sources(${LIB_NAME} PRIVATE ${PROJECT_SOURCES})

file(GLOB_RECURSE INTERNAL_HEADERS CONFIGURE_DEPENDS
"${CMAKE_CURRENT_SOURCE_DIR}/include/${CURRENT_FOLDER_NAME}/*.h"
"${CMAKE_CURRENT_SOURCE_DIR}/include/${CURRENT_FOLDER_NAME}/*.hpp"
)
list(REMOVE_ITEM INTERNAL_HEADERS
"${CMAKE_CURRENT_SOURCE_DIR}/include/${CURRENT_FOLDER_NAME}/${CURRENT_FOLDER_NAME}.h"
)
target_sources(${LIB_NAME} PRIVATE ${INTERNAL_HEADERS})

target_include_directories(${LIB_NAME}
PUBLIC
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
$<INSTALL_INTERFACE:include>
)

message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}/include/${CURRENT_FOLDER_NAME}/pch.h")
target_precompile_headers(${LIB_NAME}
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include/${CURRENT_FOLDER_NAME}/pch.h)

target_include_directories(${LIB_NAME} PRIVATE ${nanobind_SOURCE_DIR}/include)
target_link_libraries(${LIB_NAME} PRIVATE Base)
target_link_libraries(${LIB_NAME} PRIVATE Measurement)

install(TARGETS ${LIB_NAME}
    LIBRARY DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    RUNTIME DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    ARCHIVE DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)