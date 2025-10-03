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
target_link_libraries(${LIB_NAME} PRIVATE Measurements)

install(TARGETS ${LIB_NAME}
    LIBRARY DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    RUNTIME DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    ARCHIVE DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

set_target_properties(${LIB_NAME} PROPERTIES
  PREFIX ""                                      # remove lib prefix on unix so import name matches
  LIBRARY_OUTPUT_DIRECTORY "${PYTHON_PACKAGE_INSTALL_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${PYTHON_PACKAGE_INSTALL_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${PYTHON_PACKAGE_INSTALL_DIR}"
  INSTALL_RPATH "\$ORIGIN/.libs"                 # use $ORIGIN so wheel is relocatable
  BUILD_WITH_INSTALL_RPATH ON
)

# ensure extension ends up in package path inside staged install
set(PACKAGE_PY_PATH "quaos/core/light")
set(PYTHON_PACKAGE_INSTALL_DIR "${CMAKE_BINARY_DIR}/install/${PACKAGE_PY_PATH}")

# direct build output into the install staging folder so you can inspect it
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY        "${PYTHON_PACKAGE_INSTALL_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY        "${PYTHON_PACKAGE_INSTALL_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY        "${PYTHON_PACKAGE_INSTALL_DIR}")

# for multi-config generators
foreach(cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
  set_property(GLOBAL PROPERTY CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg} "${PYTHON_PACKAGE_INSTALL_DIR}")
  set_property(GLOBAL PROPERTY CMAKE_LIBRARY_OUTPUT_DIRECTORY_${cfg} "${PYTHON_PACKAGE_INSTALL_DIR}")
  set_property(GLOBAL PROPERTY CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${cfg} "${PYTHON_PACKAGE_INSTALL_DIR}")
endforeach()

# install into the Python package path (scikit-build core will stage this under site-packages)
install(TARGETS ${LIB_NAME}
        LIBRARY DESTINATION ${PACKAGE_PY_PATH}
        RUNTIME DESTINATION ${PACKAGE_PY_PATH})
        


