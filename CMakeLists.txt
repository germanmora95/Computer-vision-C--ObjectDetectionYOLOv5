
cmake_minimum_required(VERSION 3.10)
project(Cpp_YOLOv5)
find_package(OpenCV REQUIRED)
add_executable(Cpp_YOLOv5 ObjDetection_Webcam.cpp)
target_link_libraries(Cpp_YOLOv5 ${OpenCV_LIBS})