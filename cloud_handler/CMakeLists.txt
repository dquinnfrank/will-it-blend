cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project (cloud_handler)

enable_language(CXX)

# HDF5
find_package (HDF5)
INCLUDE_DIRECTORIES (${HDF5_INCLUDE_DIR})
set (LINK_LIBS ${LINK_LIBS} hdf5 hdf5_cpp)

# PCL
find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Build
set (main_file cloud)

add_executable (${main_file} ${PROJECT_SOURCE_DIR}/${main_file}.cpp)

target_link_libraries (${example} ${LINK_LIBS} ${PCL_LIBRARIES})
