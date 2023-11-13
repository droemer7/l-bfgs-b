cmake_minimum_required(VERSION 3.16)

# Define project name
project(lbfgsb_test)

# Find dependencies
find_package(Eigen3 3.4 REQUIRED)

# Set C++ flags for each build type
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -Wpedantic -O1 -g")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Wall -Wextra -Wpedantic -Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Wall -Wextra -Wpedantic -march=native -O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -Wpedantic -march=native -O3 -DNDEBUG")

# Set the build type
set(CMAKE_BUILD_TYPE "Release")

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Download GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Include GoogleTest module functions
include(GoogleTest)

# Point to main include directory and test include directory
include_directories(
  ${PROJECT_SOURCE_DIR}/../include
  ${PROJECT_SOURCE_DIR}/include
)

# Add test executable
add_executable(${PROJECT_NAME} src/test.cpp)

# Link test executable to gtest_main to define entry point
target_link_libraries(${PROJECT_NAME} gtest_main)

# Discover all tests at build time
gtest_discover_tests(${PROJECT_NAME})