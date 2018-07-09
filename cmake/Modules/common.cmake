find_package(GTest)

include_directories(
  ${GTEST_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${COMPUTECPP_PACKAGE_ROOT_DIR}/include
)

link_directories(${GTEST_LIBRARIES} ${COMPUTECPP_LIBS})

link_libraries(
  #gtest libs
  ${GTEST_BOTH_LIBRARIES}
  #sycl libs
  ${COMPUTECPP_RUNTIME_LIBRARY}
)
