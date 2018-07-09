# include common configs
include(common)

project(visioncpp-Tests CXX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test)
enable_testing()

add_custom_target(testgen COMMAND cd ${PROJECT_SOURCE_DIR}/tests/operators && ${PYTHON_EXECUTABLE} cc-gen.py)

file(GLOB_RECURSE _srcs
    ${CMAKE_BINARY_DIR}/autogen/*/*.cpp
)
# for each cc test in folder
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
  add_executable(${filename} ${_src})
  add_sycl_to_target(
    TARGET ${filename} 
    SOURCES ${_src})
  target_link_libraries(${filename} PRIVATE Threads::Threads)

  add_test(NAME ${filename} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/test/${filename})
endforeach(_src ${test_SRC})
