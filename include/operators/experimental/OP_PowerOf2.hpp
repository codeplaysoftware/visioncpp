// This file is part of VisionCPP, a lightweight C++ template library
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

/// \file OP_PowerOf2.hpp
/// \brief It calculates the power of 2 of a matrix

namespace visioncpp {
/// \struct OP_PowerOf2
/// \brief This functor implements the power of 2 of one matrix
struct OP_PowerOf2 {
  /// \brief It calculates the power of 2 ( t^2 )
  /// \param t - Float matrix
  /// \return float - Return the power of 2 ( t^2 )
  template <typename T>
  auto operator()(T t) -> decltype(t* t) {
    return t * t;
  }
};
}
