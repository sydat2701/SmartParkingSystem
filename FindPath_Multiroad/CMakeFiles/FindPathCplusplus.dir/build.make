# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/linhdt/Desktop/sydat/DATN/Yolov5_DeepSort_OSNet/FindPath_Multiroad

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/linhdt/Desktop/sydat/DATN/Yolov5_DeepSort_OSNet/FindPath_Multiroad

# Include any dependencies generated for this target.
include CMakeFiles/FindPathCplusplus.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/FindPathCplusplus.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/FindPathCplusplus.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FindPathCplusplus.dir/flags.make

CMakeFiles/FindPathCplusplus.dir/utils/BFS_pakaged.cpp.o: CMakeFiles/FindPathCplusplus.dir/flags.make
CMakeFiles/FindPathCplusplus.dir/utils/BFS_pakaged.cpp.o: utils/BFS_pakaged.cpp
CMakeFiles/FindPathCplusplus.dir/utils/BFS_pakaged.cpp.o: CMakeFiles/FindPathCplusplus.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/linhdt/Desktop/sydat/DATN/Yolov5_DeepSort_OSNet/FindPath_Multiroad/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FindPathCplusplus.dir/utils/BFS_pakaged.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/FindPathCplusplus.dir/utils/BFS_pakaged.cpp.o -MF CMakeFiles/FindPathCplusplus.dir/utils/BFS_pakaged.cpp.o.d -o CMakeFiles/FindPathCplusplus.dir/utils/BFS_pakaged.cpp.o -c /home/linhdt/Desktop/sydat/DATN/Yolov5_DeepSort_OSNet/FindPath_Multiroad/utils/BFS_pakaged.cpp

CMakeFiles/FindPathCplusplus.dir/utils/BFS_pakaged.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FindPathCplusplus.dir/utils/BFS_pakaged.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/linhdt/Desktop/sydat/DATN/Yolov5_DeepSort_OSNet/FindPath_Multiroad/utils/BFS_pakaged.cpp > CMakeFiles/FindPathCplusplus.dir/utils/BFS_pakaged.cpp.i

CMakeFiles/FindPathCplusplus.dir/utils/BFS_pakaged.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FindPathCplusplus.dir/utils/BFS_pakaged.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/linhdt/Desktop/sydat/DATN/Yolov5_DeepSort_OSNet/FindPath_Multiroad/utils/BFS_pakaged.cpp -o CMakeFiles/FindPathCplusplus.dir/utils/BFS_pakaged.cpp.s

# Object files for target FindPathCplusplus
FindPathCplusplus_OBJECTS = \
"CMakeFiles/FindPathCplusplus.dir/utils/BFS_pakaged.cpp.o"

# External object files for target FindPathCplusplus
FindPathCplusplus_EXTERNAL_OBJECTS =

FindPathCplusplus.cpython-38-x86_64-linux-gnu.so: CMakeFiles/FindPathCplusplus.dir/utils/BFS_pakaged.cpp.o
FindPathCplusplus.cpython-38-x86_64-linux-gnu.so: CMakeFiles/FindPathCplusplus.dir/build.make
FindPathCplusplus.cpython-38-x86_64-linux-gnu.so: CMakeFiles/FindPathCplusplus.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/linhdt/Desktop/sydat/DATN/Yolov5_DeepSort_OSNet/FindPath_Multiroad/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module FindPathCplusplus.cpython-38-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FindPathCplusplus.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FindPathCplusplus.dir/build: FindPathCplusplus.cpython-38-x86_64-linux-gnu.so
.PHONY : CMakeFiles/FindPathCplusplus.dir/build

CMakeFiles/FindPathCplusplus.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FindPathCplusplus.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FindPathCplusplus.dir/clean

CMakeFiles/FindPathCplusplus.dir/depend:
	cd /home/linhdt/Desktop/sydat/DATN/Yolov5_DeepSort_OSNet/FindPath_Multiroad && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/linhdt/Desktop/sydat/DATN/Yolov5_DeepSort_OSNet/FindPath_Multiroad /home/linhdt/Desktop/sydat/DATN/Yolov5_DeepSort_OSNet/FindPath_Multiroad /home/linhdt/Desktop/sydat/DATN/Yolov5_DeepSort_OSNet/FindPath_Multiroad /home/linhdt/Desktop/sydat/DATN/Yolov5_DeepSort_OSNet/FindPath_Multiroad /home/linhdt/Desktop/sydat/DATN/Yolov5_DeepSort_OSNet/FindPath_Multiroad/CMakeFiles/FindPathCplusplus.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FindPathCplusplus.dir/depend

