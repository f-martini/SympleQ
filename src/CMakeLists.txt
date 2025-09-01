cmake_minimum_required(VERSION 3.21)
project(quaos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OUTPUT_REPO)

add_subdirectory("./quaos/core/light/src")