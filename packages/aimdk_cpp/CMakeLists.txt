cmake_minimum_required(VERSION 3.16)
project(${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(aimdk_msgs REQUIRED)

python_add_library(aimdk_cpp MODULE
  src/py_binding.cpp
  src/aimdk_controller.cpp
  WITH_SOABI
)

target_link_libraries(aimdk_cpp
  PRIVATE
    pybind11::headers
    rclcpp::rclcpp
    sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
    aimdk_msgs::aimdk_msgs__rosidl_typesupport_cpp)

target_include_directories(aimdk_cpp PRIVATE src)
target_compile_definitions(aimdk_cpp PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS aimdk_cpp DESTINATION aimdk_cpp)
