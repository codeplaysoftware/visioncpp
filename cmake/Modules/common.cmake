find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  ${GTEST_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${COMPUTECPP_PACKAGE_ROOT_DIR}/include
)

link_directories(${GTEST_LIBRARIES} ${COMPUTECPP_LIBS})

link_libraries(
  #gtest libs
  ${GTEST_BOTH_LIBRARIES}
  #opencv libs
  ${OpenCV_LIBS}
  #sycl libs
  ${COMPUTECPP_RUNTIME_LIBRARY}
)
