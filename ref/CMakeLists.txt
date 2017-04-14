cmake_minimum_required(VERSION 2.8)

project(flow_ref)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_PREFIX_PATH "/usr/local/opt/opencv3/share/OpenCV")

set(OpenCV_INCLUDE_DIRS "/usr/local/opt/opencv3/include")

find_package(OpenCV 3 REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBRARIES}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${OpenCV_INCLUDE_DIRS} "inc")

# Samples
add_executable(flow_ref flow_ref.cpp)
target_link_libraries(flow_ref ${OpenCV_LIBRARIES})
