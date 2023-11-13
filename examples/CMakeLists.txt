cmake_minimum_required(VERSION 3.16)

# Define project name
project(lbfgsb_example)

# Find dependencies
find_package(Eigen3 3.4 REQUIRED)

# Set C++ flags for each build type
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -Wpedantic -O1 -g")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Wall -Wextra -Wpedantic -Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Wall -Wextra -Wpedantic -march=native -O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -Wpedantic -march=native -O3 -DNDEBUG")

# Set the build type
set(CMAKE_BUILD_TYPE "Release")

# Require at least C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to main include directory and example include directory
include_directories(
  ${PROJECT_SOURCE_DIR}/../include
  ${PROJECT_SOURCE_DIR}/include
)

# Add example executables
add_executable(${PROJECT_NAME}_constrained src/example_constrained.cpp)
add_executable(${PROJECT_NAME}_unconstrained src/example_unconstrained.cpp)
add_executable(${PROJECT_NAME}_callback src/example_callback.cpp)