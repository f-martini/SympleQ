cmake_minimum_required(VERSION 3.21)
project(quaos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OUTPUT_REPO)

message(STATUS "CMake preset: $ENV{CMAKE_GENERATOR_PRESET}")
message(STATUS "Adding: ${CMAKE_SRC_DIRECTORY}")
message(STATUS "Building for: ${CMAKE_BUILD_TYPE}")

add_subdirectory("${CMAKE_SRC_DIRECTORY}")