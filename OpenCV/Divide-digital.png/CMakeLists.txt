cmake_minimum_required(VERSION 3.24)
project(Divide_digital_png)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

add_executable(Divide_digital_png main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${InferenceEngine_LIBS})