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

/// \file OP_RGBToGREY.hpp
/// \brief it converts RGB pixel to Grey pixel

namespace visioncpp {
/// \brief This functor performs RGB to GREY convertion following rule:
/// GREY <- 0.299f * R + 0,587f * G + 0.114 * B
struct OP_RGBToGREY {
  /// \param in - RGB pixel.
  /// \returns float - greyscale value.
  float operator()(visioncpp::pixel::F32C3 in) {
    //  because it is saved cv in our rgb format then b stored in r
    // and r stored in b.
    // luminance , the most accurate one
    return 0.299f * in[0] + 0.587f * in[1] + 0.114f * in[2];
  }
};
}
