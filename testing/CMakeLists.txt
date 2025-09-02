cmake_minimum_required(VERSION 3.10)
project(FosterML_Tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Paths
set(PROJECT_ROOT ${CMAKE_SOURCE_DIR}/..)
set(SRC_DIR ${PROJECT_ROOT}/src)
set(MAIN_INCLUDE_DIR ${PROJECT_ROOT}/include)

# Choose test folder
set(TEST_NAME "" CACHE STRING "Name of the test folder to build, e.g., '02_Tensor'")
if(NOT TEST_NAME)
    message(FATAL_ERROR "Please specify -DTEST_NAME=... to select a test folder")
endif()
set(TEST_DIR ${CMAKE_SOURCE_DIR}/${TEST_NAME})
set(MAIN_CPP ${TEST_DIR}/main.cpp)

if(NOT EXISTS ${MAIN_CPP})
    message(FATAL_ERROR "main.cpp not found in test folder: ${TEST_DIR}")
endif()

# Include local headers first, then main headers
include_directories(
    ${TEST_DIR}          # local headers/templates override main ones
    ${MAIN_INCLUDE_DIR}  # main project headers
)

# Core project sources (non-template .cpp files only)
file(GLOB LIB_SOURCES ${SRC_DIR}/*.cpp)

# Sanitize target name (no spaces)
string(REPLACE " " "_" TEST_TARGET ${TEST_NAME})

# Create executable
add_executable(${TEST_TARGET} ${LIB_SOURCES} ${MAIN_CPP})

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(${TEST_TARGET} PRIVATE -fsanitize=address -fno-omit-frame-pointer -g)
    target_link_options(${TEST_TARGET} PRIVATE -fsanitize=address)
endif()
