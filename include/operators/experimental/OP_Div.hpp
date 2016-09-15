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

/// \file OP_Div.hpp
/// \brief It divides two matrices element-wise

namespace visioncpp {
/// \brief This functor divides two matrices element-wise
struct OP_Div {
  /// \brief This functor divides two matrices element-wise
  /// \param t1 - Matrix1
  /// \param t2 - Matrix2
  /// \return T - Division ( t1 / t2 )
  template <typename T1, typename T2>
  auto operator()(T1 t1, T2 t2) -> decltype(t1 / t2) {
    return t1 / t2;
  }
};
}
