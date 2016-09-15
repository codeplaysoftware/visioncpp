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

/// \file OP_HSVToU8C3.hpp
/// \brief it converts HSV pixel to U8C3 pixel

namespace visioncpp {
/// \brief This functor allows displaying HSV.
/// the formula followed for unsigned char: 255S, 255V, 128H
struct OP_HSVToU8C3 {
  /// \param in
  /// \return U8C3
  visioncpp::pixel::U8C3 operator()(visioncpp::pixel::F32C3 in) {
    const float FLOAT_TO_BYTE = 255.0f;

    // Convert from floats to 8-bit integers
    // let follow OpenCV convention of
    // V <- 255V, S <- 255S, H <- H/2(to fit 0 to 255)
    float bH = (float)(in[0] * 180.0f);
    float bS = (float)(in[1] * FLOAT_TO_BYTE);
    float bV = (float)(in[2] * FLOAT_TO_BYTE);

    return visioncpp::pixel::U8C3(static_cast<unsigned char>(bH),
                                  static_cast<unsigned char>(bS),
                                  static_cast<unsigned char>(bV));
  }
};
}
