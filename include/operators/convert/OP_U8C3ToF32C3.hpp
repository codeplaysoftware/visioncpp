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

/// \file OP_U8C3ToF32C3.hpp
/// \brief it converts U8C3 pixel to F32C3 pixel

namespace visioncpp {
/// \brief This functor performs conversion from [0, 255] to [0.0f, 1.0f]
struct OP_U8C3ToF32C3 {
  /// \param in - three-channel unsigned char
  /// \return F32C3 - three-channel float
  visioncpp::pixel::F32C3 operator()(visioncpp::pixel::U8C3 in) {
    const float FLOAT_TO_BYTE = 255.0f;
    const float BYTE_TO_FLOAT = 1.0f / FLOAT_TO_BYTE;
    return visioncpp::pixel::F32C3(static_cast<float>(in[0] * BYTE_TO_FLOAT),
                                   static_cast<float>(in[1] * BYTE_TO_FLOAT),
                                   static_cast<float>(in[2] * BYTE_TO_FLOAT));
  }
};
}
