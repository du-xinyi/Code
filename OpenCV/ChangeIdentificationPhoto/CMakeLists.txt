cmake_minimum_required(VERSION 3.24)
project(ChangeIdentificationPhoto)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

add_executable(ChangeIdentificationPhoto main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})