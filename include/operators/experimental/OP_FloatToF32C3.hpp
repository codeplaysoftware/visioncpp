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

/// \file OP_FloatToF32C3.hpp
/// \brief Converts from uchar to float and float to uchar

namespace visioncpp {

/// \brief It replicates one channel to 3 channels
struct OP_FloatToF32C3 {
  /// \brief It replicates one channel to 3 channels
  /// \param t - Float of 1 channel
  /// \return F32C3 - Float of 3 channels
  visioncpp::pixel::F32C3 operator()(float t) {
    return visioncpp::pixel::F32C3(t, t, t);
  }
};

/// \brief It converts float to uchar converting [0.0f, 1.0f] to [0, 255]
struct OP_FloatToU8C1 {
  /// \brief It converts float to uchar converting [0.0f, 1.0f] to [0, 255]
  /// \param t - float of 1 channel
  /// \return U8C1 - uchar of 1 channel
  visioncpp::pixel::U8C1 operator()(float t) {
    return visioncpp::pixel::U8C1(static_cast<unsigned char>(t * 255));
  }
};

/// \brief It converts uchar to float converting range [0, 255] to [0.0f, 1.0f]
struct OP_U8C1ToFloat {
  /// \brief It converts uchar to float converting range [0, 255] to [0.0f, 1.0f]
  /// \param t - uchar of 1 channel
  /// \return float - float of 1 channels
  float operator()(visioncpp::pixel::U8C1 t) {
    return static_cast<float>(t[0]) / 255.0f;
  }
};

/// \brief It converts float to uchar
struct OP_FloatToUChar {
  /// \brief It converts float to uchar
  /// \param t - float of 1 channel
  /// \return F32C3 - uchar of 1 channel
  unsigned char operator()(float t) {
    return (static_cast<unsigned char>(t * 255));
  }
};
}
