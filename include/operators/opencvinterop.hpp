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

/// \file opencvinterop.hpp
/// \brief This header gathers all interoperability with OpenCV operations.

namespace visioncpp {
/// \struct OP_CVBGRToRGB
/// \brief This node is a utility node that does a conversion from cv::Mat (
/// unsigned char storage; channel order BGR ) to float with order of RGB
/// normalise from opencv format to VisionCpp internal format [0, 255] to
/// [0.0f, 1.0f]
struct OP_CVBGRToRGB {
  /// \param in
  /// \return F32C3
  visioncpp::pixel::F32C3 operator()(visioncpp::pixel::U8C3 in) {
    visioncpp::pixel::F32C3 out(static_cast<float>(in[2] / 255.0f),
                                static_cast<float>(in[1] / 255.0f),
                                static_cast<float>(in[0] / 255.0f));
    return out;
  }
};

/// \struct OP_RGBToCVBGR
/// \brief This node is a utility node that does a conversion from float with
/// order of RGB to  cv::Mat ( unsigned char storage; channel order BGR )
/// denormalise from VisionCpp base format ( \ref F32C3 ) to opencv 8UC3
/// (three channel unsigned char)
struct OP_RGBToCVBGR {
  /// \param in
  /// \return U8C3
  visioncpp::pixel::U8C3 operator()(visioncpp::pixel::F32C3 in) {
    visioncpp::pixel::U8C3 out(static_cast<unsigned char>(in[2] * 255.0f),
                               static_cast<unsigned char>(in[1] * 255.0f),
                               static_cast<unsigned char>(in[0] * 255.0f));
    return out;
  }
};

/// \struct OP_GREYToCVBGR
/// \brief float between [0.0f, 1.0f] to [0, 255]
/// One channel GREY ( float ) is going to be converted to one channel cv::Mat
struct OP_GREYToCVBGR {
  /// \param in
  /// \return U8C1
  unsigned char operator()(float in) {
    return static_cast<unsigned char>(in * 255);
  }
};
}
