
cmake_minimum_required(VERSION 3.5)
project(PLateDetect)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
find_package(OpenCV 3.4.0 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories(include) 
include_directories(../include /usr/local/include /usr/local/cuda/include /usr/include)
include_directories(/usr/include/freetype2/)

set(SRC_DETECTION src/Plate_Detect.cpp include/Plate_Detect.h)
set(SRC_RECOGNIZER src/PlateRecognizer.cpp )
set(SRC_PIPLINE src/Pipeline.cpp include/Pipeline.h)
set(SRC_FASTDESKEW src/FastDeskew.cpp )
set(CVTEXT src/CvxText.cpp include/CvxText.h)
LINK_DIRECTORIES(../build/lib)

add_executable(Detect ${CVTEXT} ${SRC_DETECTION} ${SRC_FASTDESKEW} ${SRC_RECOGNIZER} ${SRC_PIPLINE} Demo/test_pipeline.cpp)

target_link_libraries(Detect ${OpenCV_LIBS} freetype libcaffe.so /usr/lib/x86_64-linux-gnu/libboost_system.so /usr/lib/x86_64-linux-gnu/libglog.so)
