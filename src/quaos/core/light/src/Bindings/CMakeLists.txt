
set(LIB_NAME "light")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(nanobind)

find_program(NVCC_EXECUTABLE nvcc HINTS "$ENV{CUDA_PATH}/bin")
if(NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found! Please set CUDA_PATH or install CUDA.")
endif()
message(STATUS "Found nvcc at ${NVCC_EXECUTABLE}")

find_library(CUDA_CUDART_LIBRARY cudart HINTS "$ENV{CUDA_PATH}/lib/x64")
if(NOT CUDA_CUDART_LIBRARY)
    message(FATAL_ERROR "Could not find cudart library")
endif()
message(STATUS "Found cudart library: ${CUDA_CUDART_LIBRARY}")

file(GLOB CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
set(CUDA_LIB "${CMAKE_BINARY_DIR}/kernels_static.lib")

add_custom_command(
    OUTPUT ${CUDA_LIB}
    COMMAND "${NVCC_EXECUTABLE}" -lib ${CUDA_SOURCES} -o ${CUDA_LIB}
            --compiler-options "/MD" 
    DEPENDS ${CUDA_SOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    COMMENT "Building CUDA static library"
    VERBATIM
)

add_custom_target(kernels_static ALL DEPENDS ${CUDA_LIB})

nanobind_add_module(${LIB_NAME} bindings.cpp)
target_link_libraries(${LIB_NAME} PRIVATE ${CUDA_LIB} ${CUDA_CUDART_LIBRARY})

install(TARGETS ${LIB_NAME}
    LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    RUNTIME DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    ARCHIVE DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../.."
)