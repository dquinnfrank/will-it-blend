cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project (cloud_handler)

# Put all output into a bin directory
#set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Get header files from include directory
#include_directories(${PROJECT_SOURCE_DIR}/include)

# Set the names of inputs and outputs
set (main_file cloud_old)
set (out_name cloud_handler)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#SET(CMAKE_SWIG_OUTDIR ${PROJECT_SOURCE_DIR}/bin)

enable_language(CXX)

# Get swig
#FIND_PACKAGE(SWIG REQUIRED)
#INCLUDE(${SWIG_USE_FILE})

# Get python
#FIND_PACKAGE(PythonLibs)
#INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
#INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})

# Basics
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
#INCLUDE_DIRECTORIES("..")

# Flags, C++ and include numpy
#SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-I/usr/lib/python2.7/dist-packages/numpy/core/include")

# CMAKE_SWIG_OUTDIR isn't working, fix it here
#SET(CMAKE_SWIG_FLAGS "-outdir ${PROJECT_SOURCE_DIR}/bin")
#SET(CMAKE_SWIG_FLAGS "-outcurrentdir")

# HDF5
find_package (HDF5 REQUIRED)
include_directories (${HDF5_INCLUDE_DIR})
set (LINK_LIBS ${LINK_LIBS} hdf5 hdf5_cpp)

# PCL
find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS}) 

# EIGEN
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

 # Use c++ compilier
#SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/${main_file}.i PROPERTIES CPLUSPLUS ON)
#SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/include/${main_file}.i PROPERTIES SWIG_FLAGS "-includeall") # This breaks iostream for some reason

# Builds the normal cpp executable
add_executable (${out_name} ${PROJECT_SOURCE_DIR}/${main_file}.cpp)

target_link_libraries (${out_name} ${LINK_LIBS} ${PCL_LIBRARIES})

# Build SWIG bindings
#SWIG_ADD_MODULE(${out_name} python ${PROJECT_SOURCE_DIR}/${main_file}.i ${PROJECT_SOURCE_DIR}/${main_file}.cpp)

# No idea what this does
#SWIG_LINK_LIBRARIES(${out_name} ${PYTHON_LIBRARIES})
