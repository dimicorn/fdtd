cmake_minimum_required(VERSION 3.24)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
project(fdtd1d CXX)
set(CMAKE_CXX_STANDARD 17)
include_directories(include)

add_executable(fdtd1d src/main.cpp src/fdtd1d.cpp)

include(FetchContent)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(test_fdtd1d tests/test_fdtd1d.cpp src/fdtd1d.cpp)
target_link_libraries(test_fdtd1d GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_fdtd1d)
