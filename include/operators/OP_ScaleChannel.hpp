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

/// \file OP_ScaleChannel.hpp
/// \brief This files contains structs that change different channels by factor
/// passed via float f.

namespace visioncpp {
/// \struct OP_ScaleChannelZero
/// \brief This custom functor changes 0 channel by factor passed via float f.
struct OP_ScaleChannelZero {
  /// \param lhs - Pixel in HSV color space
  /// \param f - S channel multiplier
  /// \return F32C3 - Altered HSV color space pixel
  visioncpp::pixel::F32C3 operator()(visioncpp::pixel::F32C3 lhs, float f) {
    lhs[0] *= f;
    return lhs;
  }
};

/// \struct OP_ScaleChannelOne
/// \brief This custom functor changes 1 channel by factor passed via float f.
struct OP_ScaleChannelOne {
  /// \param lhs - Pixel in HSV color space
  /// \param f - S channel multiplier
  /// \return F32C3 - Altered HSV color space pixel
  visioncpp::pixel::F32C3 operator()(visioncpp::pixel::F32C3 lhs, float f) {
    lhs[1] *= f;
    return lhs;
  }
};

/// \struct OP_ScaleChannelTwo
/// \brief This custom functor changes 2 channel by factor passed via float f.
struct OP_ScaleChannelTwo {
  /// \param lhs - Pixel in HSV color space
  /// \param f - S channel multiplier
  /// \return F32C3 - Altered HSV color space pixel
  visioncpp::pixel::F32C3 operator()(visioncpp::pixel::F32C3 lhs, float f) {
    lhs[2] *= f;
    return lhs;
  }
};
}
