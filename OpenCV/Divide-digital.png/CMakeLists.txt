cmake_minimum_required(VERSION 3.17)
project(Divide-digital.png)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

set(CMAKE_PREFIX_PATH /opt/intel/openvino_2021/deployment_tools/inference_engine/share)
find_package(InferenceEngine)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${InferenceEngine_LIBS})
