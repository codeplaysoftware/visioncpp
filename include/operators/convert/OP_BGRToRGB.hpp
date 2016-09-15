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

/// \file OP_BGRToRGB.hpp
/// \brief it converts BGR pixel to RGB pixel

namespace visioncpp {
/// \brief This functor reorders channels BGR to RGB
struct OP_BGRToRGB {
  /// \brief functor that reorders channels, e.g. HSV becomes VSH
  /// \param in - three channel unsigned char
  /// \return U8C3 - three channel unsigned char
  visioncpp::pixel::U8C3 operator()(visioncpp::pixel::U8C3 in) {
    return visioncpp::pixel::U8C3(static_cast<unsigned char>(in[2]),
                                  static_cast<unsigned char>(in[1]),
                                  static_cast<unsigned char>(in[0]));
  }
};
}
