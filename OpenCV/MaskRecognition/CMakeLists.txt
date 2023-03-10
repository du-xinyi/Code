cmake_minimum_required(VERSION 3.24)
project(MaskRecognition)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

add_executable(MaskRecognition main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})