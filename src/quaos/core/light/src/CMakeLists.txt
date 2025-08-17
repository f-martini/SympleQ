find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# --- Fetch nanobind ---
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(nanobind)

# --- Find NVCC ---
find_program(NVCC_EXECUTABLE nvcc HINTS "$ENV{CUDA_PATH}/bin")
if(NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found! Please set CUDA_PATH or install CUDA.")
endif()
message(STATUS "Found nvcc at ${NVCC_EXECUTABLE}")

# --- Find CUDA Runtime ---
find_library(CUDA_CUDART_LIBRARY cudart HINTS "$ENV{CUDA_PATH}/lib/x64")
if(NOT CUDA_CUDART_LIBRARY)
    message(FATAL_ERROR "Could not find cudart library")
endif()
message(STATUS "Found cudart library: ${CUDA_CUDART_LIBRARY}")

# --- CUDA kernel sources ---
file(GLOB CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu")

set(CUDA_LIB "${CMAKE_BINARY_DIR}/kernels_static.lib")

# --- Custom command to build CUDA static lib ---
add_custom_command(
    OUTPUT ${CUDA_LIB}
    COMMAND "${NVCC_EXECUTABLE}" -lib ${CUDA_SOURCES} -o ${CUDA_LIB}
            --compiler-options "/MD"  # match MSVC runtime
    DEPENDS ${CUDA_SOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    COMMENT "Building CUDA static library"
    VERBATIM
)

# --- Dummy target for the CUDA lib ---
add_custom_target(kernels_static ALL DEPENDS ${CUDA_LIB})

# --- Nanobind Python extension (pure C++) ---
nanobind_add_module(light bindings.cpp)
target_link_libraries(light PRIVATE ${CUDA_LIB} ${CUDA_CUDART_LIBRARY})

install(TARGETS light
    LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/.."
    RUNTIME DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/.."
    ARCHIVE DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/.."
)