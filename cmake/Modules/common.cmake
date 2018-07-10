find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(
  ${GTEST_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${ComputeCPP_DIR}/include
)

link_directories(${GTEST_LIBRARIES} ${COMPUTECPP_LIBS})

link_libraries(
  #opencv libs
  ${OpenCV_LIBS}
  #gtest libs
  ${GTEST_BOTH_LIBRARIES}
  #sycl libs
  ${COMPUTECPP_RUNTIME_LIBRARY}
)
