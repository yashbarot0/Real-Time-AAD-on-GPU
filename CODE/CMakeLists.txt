cmake_minimum_required(VERSION 3.15)
project(AAD)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add your source files here
# Example: set(SOURCES src/main.cpp src/foo.cpp)

add_executable(${PROJECT_NAME} main.cpp)

# Add include directories if needed
# target_include_directories(${PROJECT_NAME} PRIVATE include)