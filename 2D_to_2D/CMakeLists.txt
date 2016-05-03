cmake_minimum_required(VERSION 2.8)
project(2D)
find_package(OpenCV)
set(target 2D_to_2D)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(matcher_src
	../odometry/src/Matcher.cpp
	../odometry/src/Odometry.cpp
)

set(matcher_headers
	../odometry/include/Matcher.h
	../odometry/include/Odometry.h
)
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ../odometry/include)
add_executable(${target} src/2D_to_2D.cpp ${matcher_src} ${matcher_headers})
target_link_libraries(${target} ${OpenCV_LIBS})
