cmake_minimum_required(VERSION 2.8)
project(VisualOdometry)
find_package(OpenCV)
set(target viso)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(matcher_src
	src/Odometry.cpp
	src/Matcher.cpp
	src/Scale.cpp
)

set(matcher_headers
	include/Odometry.h
	include/Matcher.h
	include/Scale.h
)
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_executable(${target} src/main_viso.cpp ${matcher_src} ${matcher_headers})
target_link_libraries(${target} ${OpenCV_LIBS})
