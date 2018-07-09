# include common configs
include(common)

project(visioncpp-Examples CXX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/example)

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

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
  target_link_libraries(example_${filename} PRIVATE Threads::Threads)

  add_sycl_to_target(
    TARGET example_${filename} 
    SOURCES ${_src})

endforeach(_src ${_srcs})
