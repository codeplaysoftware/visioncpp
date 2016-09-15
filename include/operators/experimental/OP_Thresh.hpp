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

/// \file OP_Thresh.hpp
/// \brief it applies a threshold operation in the image

namespace visioncpp {
/// \struct OP_Thresh
/// \brief Implements a binary threshold
struct OP_Thresh {
  /// \brief This functor implements a binary threshold
  /// \param t1 - Image
  /// \param thresh - float threshold value
  /// \return U8C1 - Returns a binary image (1 if greater than threshold, 0
  /// otherwise)
  template <typename T1, typename T2>
  visioncpp::pixel::U8C1 operator()(T1 t1, T2 thresh) {
    return t1 > thresh ? 1 : 0;
  }
};
}
