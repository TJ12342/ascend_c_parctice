# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_COMMAND = /home/ma-user/work/cmake-3.28.3-linux-aarch64/bin/cmake

# The command to remove a file.
RM = /home/ma-user/work/cmake-3.28.3-linux-aarch64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ma-user/work/TJ111/Pdist

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ma-user/work/TJ111/Pdist/build_out

# Utility rule file for binary.

# Include any custom commands dependencies for this target.
include op_kernel/CMakeFiles/binary.dir/compiler_depend.make

# Include the progress variables for this target.
include op_kernel/CMakeFiles/binary.dir/progress.make

binary: op_kernel/CMakeFiles/binary.dir/build.make
.PHONY : binary

# Rule to build all files generated by this target.
op_kernel/CMakeFiles/binary.dir/build: binary
.PHONY : op_kernel/CMakeFiles/binary.dir/build

op_kernel/CMakeFiles/binary.dir/clean:
	cd /home/ma-user/work/TJ111/Pdist/build_out/op_kernel && $(CMAKE_COMMAND) -P CMakeFiles/binary.dir/cmake_clean.cmake
.PHONY : op_kernel/CMakeFiles/binary.dir/clean

op_kernel/CMakeFiles/binary.dir/depend:
	cd /home/ma-user/work/TJ111/Pdist/build_out && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ma-user/work/TJ111/Pdist /home/ma-user/work/TJ111/Pdist/op_kernel /home/ma-user/work/TJ111/Pdist/build_out /home/ma-user/work/TJ111/Pdist/build_out/op_kernel /home/ma-user/work/TJ111/Pdist/build_out/op_kernel/CMakeFiles/binary.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : op_kernel/CMakeFiles/binary.dir/depend

