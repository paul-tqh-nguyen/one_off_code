cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(pcap_reader VERSION 0.1.0)

# Include dir
include_directories(/usr/local/include)

# Src
AUX_SOURCE_DIRECTORY(src SRC_FILES)

# Headers
set(PROJECT_SOURCE_DIR "src")
set(PROJECT_INCLUDE_DIR "include")

# Source files
set(SOURCE_FILES
    ${PROJECT_INCLUDE_DIR}/pcap_reader.hpp
    ${PROJECT_SOURCE_DIR}/pcap_reader.cpp
)

# Set up such that XCode organizes the files correctly
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES})

# Add library
add_library(pcap_reader SHARED ${SOURCE_FILES})

# Include directories
target_include_directories(pcap_reader PRIVATE include)

# Install
install(TARGETS pcap_reader DESTINATION lib)

# Install the headers
install(FILES include/pcap_reader DESTINATION include)

# Create base directory
install(DIRECTORY include DESTINATION include)
