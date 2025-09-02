get_filename_component(CURRENT_FOLDER_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
set(LIB_NAME "light")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.8.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(nanobind)

# find_program(NVCC_EXECUTABLE nvcc HINTS "$ENV{CUDA_PATH}/bin")
# if(NOT NVCC_EXECUTABLE)
#    message(FATAL_ERROR "nvcc not found! Please set CUDA_PATH or install CUDA.")
# endif()
# message(STATUS "Found nvcc at ${NVCC_EXECUTABLE}")

# find_library(CUDA_CUDART_LIBRARY cudart HINTS "$ENV{CUDA_PATH}/lib/x64")
# if(NOT CUDA_CUDART_LIBRARY)
#     message(FATAL_ERROR "Could not find cudart library")
# endif()
# message(STATUS "Found cudart library: ${CUDA_CUDART_LIBRARY}")

# file(GLOB CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
# set(CUDA_LIB "${CMAKE_BINARY_DIR}/kernels_static.lib")

# add_custom_command(
    #     OUTPUT ${CUDA_LIB}
    #     COMMAND "${NVCC_EXECUTABLE}" -lib ${CUDA_SOURCES} -o ${CUDA_LIB}
    #             --compiler-options "/MD" 
    #     DEPENDS ${CUDA_SOURCES}
#     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/kernels
#     COMMENT "Building CUDA static library"
#     VERBATIM
# )

# add_custom_target(kernels_static ALL DEPENDS ${CUDA_LIB})


nanobind_add_module(${LIB_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp")

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

target_include_directories(${LIB_NAME} PRIVATE ${nanobind_SOURCE_DIR}/include)
target_link_libraries(${LIB_NAME} PRIVATE Base)
# target_link_libraries(${LIB_NAME} PRIVATE ${CUDA_LIB} ${CUDA_CUDART_LIBRARY})

install(TARGETS ${LIB_NAME}
    LIBRARY DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    RUNTIME DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    ARCHIVE DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)