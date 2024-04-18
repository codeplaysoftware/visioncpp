# include common configs
include(common)
if(POLICY CMP0058) 
    cmake_policy(SET CMP0058 NEW) 
endif() 

project(visioncpp-Examples CXX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/example)

file(GLOB_RECURSE _srcs ${PROJECT_SOURCE_DIR}/examples/*.cpp)
# for each cc example in folder
foreach(_src ${_srcs})

  # take folder path
  get_filename_component(path ${_src} PATH)

  # take name of file
  get_filename_component(filename ${_src} NAME_WE)

  set(inc_dir_device "")
  get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
  foreach(inc_dir_element ${inc_dirs})
    set(inc_dir_device ${inc_dir_device} -I${inc_dir_element})
  endforeach()

  # add executable that depends on the generated file
  add_executable(example_${filename} ${_src})
  # link the pthreads library to the executable
  target_link_libraries(example_${filename} 
    PRIVATE Threads::Threads)
  # link sycl with the executable
  add_sycl_to_target(
    TARGET example_${filename} 
    SOURCES ${_src})

endforeach(_src ${_srcs})
