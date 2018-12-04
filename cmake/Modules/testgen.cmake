project(testgen CXX)

execute_process(
   COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/operators/cc-gen.py --builddir ${CMAKE_BINARY_DIR} --testdir ${PROJECT_SOURCE_DIR}/tests/operators
   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/operators/
   RESULT_VARIABLE result
   OUTPUT_VARIABLE ver
 )

if(${result})
  message(FATAL_ERROR "${PYTHON_EXECUTABLE} cc-gen.py failed to generate tests.")
else()
  message(STATUS "${PYTHON_EXECUTABLE} cc-gen.py generated tests.")
endif()
