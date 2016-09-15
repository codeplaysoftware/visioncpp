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

/// \file OP_F32C3ToU8C3.hpp
/// \brief it converts F32C3 pixel to U8C3 pixel

namespace visioncpp {
/// \brief This functor performs conversion from [0.0f, 1.0f] to [0, 255]
struct OP_F32C3ToU8C3 {
  /// \param in - three-channel float
  /// \return U8C3 - three-channel unsigned char
  visioncpp::pixel::U8C3 operator()(visioncpp::pixel::F32C3 in) {
    const float FLOAT_TO_BYTE = 255.0f;
    return visioncpp::pixel::U8C3(
        static_cast<unsigned char>(in[0] * FLOAT_TO_BYTE),
        static_cast<unsigned char>(in[1] * FLOAT_TO_BYTE),
        static_cast<unsigned char>(in[2] * FLOAT_TO_BYTE));
  }
};
}
