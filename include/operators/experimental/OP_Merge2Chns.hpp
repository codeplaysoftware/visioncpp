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

/// \file OP_Merge2Chns.hpp
/// \brief It merges 2 matrices of one channels into one matrix of 2 channels

namespace visioncpp {
/// \brief This functor merges 2 matrices into one matrix of 2 channels
struct OP_Merge2Chns {
  /// \brief Merge 2 matrices into one matrix onf 2 channels
  /// \param t1 - One channel float
  /// \param t2 - One channel float
  /// \return F32C2 - 2 channels float
  template <typename T1, typename T2>
  visioncpp::pixel::F32C2 operator()(T1 t1, T2 t2) {
    return visioncpp::pixel::F32C2(t1, t2);
  }
};
}
