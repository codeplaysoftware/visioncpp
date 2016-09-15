// This file is part of VisionCpp, a lightweight C++ template library
// for computer vision and image processing.
//
// Copyright (C) 2016 Codeplay Software Limited. All Rights Reserved.
//
// Contact: visioncpp@codeplay.com
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// \file time.hpp
/// \brief Basic time operations

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_TOOLS_TIME_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_TOOLS_TIME_HPP_

#include <chrono>

namespace visioncpp {
namespace internal {
namespace tools {
/// function get_elapse_time
/// \brief it is used to calculate the time period in second.
/// template parameters
/// \tparam T : the type of the time from chrono
/// function parameters
/// \param begin : start of the time
/// \param end : end of the time
/// returns a double
template <typename T>
double get_elapse_time(T begin, T end) {
  return std::chrono::duration<double, std::chrono::seconds::period>(
             end - begin).count();
};
/// function get_current_time
/// \brief getting the current time from the system using chrono
/// \return std::chrono::high_resolution_clock::time_point
std::chrono::high_resolution_clock::time_point get_current_time() {
  return std::chrono::high_resolution_clock::now();
}
}  // tools
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_TOOLS_TIME_HPP_
