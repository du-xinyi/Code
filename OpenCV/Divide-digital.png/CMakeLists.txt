cmake_minimum_required(VERSION 3.24)
project(Divide_digital_png)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

set(CMAKE_PREFIX_PATH /opt/intel/openvino_2021/deployment_tools/inference_engine/share)
find_package(InferenceEngine)

add_executable(Divide_digital_png main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${InferenceEngine_LIBS})