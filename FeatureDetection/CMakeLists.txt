cmake_minimum_required(VERSION 2.8)
project(FeatureDetection)
find_package(OpenCV)
set(target DetectFeatures)
set(matcher_src
	src/Matcher.cpp
)

set(matcher_headers
	include/Matcher.h
)
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_executable(${target} src/main_matcher_test.cpp ${matcher_src} ${matcher_headers})
target_link_libraries(${target} ${OpenCV_LIBS})
