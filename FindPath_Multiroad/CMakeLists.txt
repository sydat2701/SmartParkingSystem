cmake_minimum_required(VERSION 3.4)
add_subdirectory(pybind11)
pybind11_add_module(FindPathCplusplus utils/BFS_pakaged.cpp)
pybind11_add_module(FillRoadCplusplus utils/FillRoad_faster.cpp)
pybind11_add_module(CheckPointInPolygonCplusplus utils/CheckPointInPolygon.cpp)
