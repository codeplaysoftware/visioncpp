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

/// \file OP_Scale.hpp
/// \brief It scales the pixel value of an image by a factor

namespace visioncpp {
/// \struct OP_Scale
/// \brief Scales the pixel value of an image by a factor
struct OP_Scale {
  /// \brief Scales each pixel of an image by a factor
  /// \param t1 - Image of one channel
  /// \param f - Scale factor
  /// \return T1 - Returns the scaled image of time of the input image
  template <typename T1, typename T2>
  T1 operator()(T1 t1, T2 f) {
    return t1 * f;
  }
};
}
