get_filename_component(CURRENT_FOLDER_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
set(LIB_NAME "light")

# Where inside the Python package the extension should be installed
set(PACKAGE_PY_PATH "quaos/core/light")
set(PYTHON_PACKAGE_INSTALL_DIR "${CMAKE_BINARY_DIR}/install/${PACKAGE_PY_PATH}")

# Place built binaries into the staged install tree so they will be picked up by scikit-build
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY        "${PYTHON_PACKAGE_INSTALL_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY        "${PYTHON_PACKAGE_INSTALL_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY        "${PYTHON_PACKAGE_INSTALL_DIR}")

# For multi-config generators (Visual Studio / Xcode)
foreach(cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
  set_property(GLOBAL PROPERTY CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg} "${PYTHON_PACKAGE_INSTALL_DIR}")
  set_property(GLOBAL PROPERTY CMAKE_LIBRARY_OUTPUT_DIRECTORY_${cfg} "${PYTHON_PACKAGE_INSTALL_DIR}")
  set_property(GLOBAL PROPERTY CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${cfg} "${PYTHON_PACKAGE_INSTALL_DIR}")
endforeach()

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

# Create the nanobind module target
nanobind_add_module(${LIB_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/light.cpp")

# Add sources and headers
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
target_link_libraries(${LIB_NAME} PRIVATE Base Measurements)

# Make import name predictable on Unix and set relocatable rpath
set_target_properties(${LIB_NAME} PROPERTIES
  PREFIX ""                                      # remove lib prefix on unix so import name matches
  INSTALL_RPATH "\$ORIGIN/.libs"                 # use $ORIGIN so wheel is relocatable
  BUILD_WITH_INSTALL_RPATH ON
)

# Install into the Python package path (scikit-build will stage this into the wheel)
install(TARGETS ${LIB_NAME}
        LIBRARY DESTINATION ${PACKAGE_PY_PATH}
        RUNTIME DESTINATION ${PACKAGE_PY_PATH}
        ARCHIVE DESTINATION ${PACKAGE_PY_PATH}
)
        


