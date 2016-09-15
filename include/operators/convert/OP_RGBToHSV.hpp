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

/// \file OP_RGBToHSV.hpp
/// \brief it converts RGB pixel to HSV pixel

namespace visioncpp {
/// \brief Functor converts RGB ( R: 0.0f..1.0f, G: 0.0f..1.0f, B: 0.0f..1.0f)
/// to HSV ( H: 0.0f..360.f, S: 0.0f..1.0f V: 0.0f..1.0f ) color space
struct OP_RGBToHSV {
  /// \param inRGB
  /// \return F32C3
  visioncpp::pixel::F32C3 operator()(visioncpp::pixel::F32C3 inRGB) {
    // Convert from RGB to HSV, using float ranges 0.0 to 1.0.
    float fR = inRGB[0], fG = inRGB[1], fB = inRGB[2];
    float fH = 0.0f, fS = 0.0f, fV = 0.0f;

    float fDelta;
    float fMin, fMax;
    // Get the min and max, but use integer comparisons for slight speedup.
    if (inRGB[2] < inRGB[1]) {
      if (inRGB[2] < inRGB[0]) {
        fMin = fB;
        if (inRGB[0] > inRGB[1]) {
          fMax = fR;
        } else {
          fMax = fG;
        }
      } else {
        fMin = fR;
        fMax = fG;
      }
    } else {
      if (inRGB[1] < inRGB[0]) {
        fMin = fG;
        if (inRGB[2] > inRGB[0]) {
          fMax = fB;
        } else {
          fMax = fR;
        }
      } else {
        fMin = fR;
        fMax = fB;
      }
    }
    fDelta = fMax - fMin;
    fV = fMax;             // Value (Brightness).
    if (fMax != 0.0f) {    // Make sure it's not pure black.
      fS = fDelta / fMax;  // Saturation.
      float ANGLE_TO_UNIT =
          1.0f /
          (6.0f * fDelta);  // Make the Hues between 0.0 to 1.0 instead of 6.0
      if (fDelta == 0.0f) {
        fH = 0.0f;                    // undefined hue
      } else if (fMax == inRGB[0]) {  // between yellow and magenta.
        fH = (fG - fB) * ANGLE_TO_UNIT;
      } else if (fMax == inRGB[1]) {  // between cyan and yellow.
        fH = (2.0f / 6.0f) + (fB - fR) * ANGLE_TO_UNIT;
      } else {  // between magenta and cyan.
        fH = (4.0f / 6.0f) + (fR - fG) * ANGLE_TO_UNIT;
      }
      // Wrap outlier Hues around the circle.
      if (fH < 0.0f) fH += 1.0f;
      if (fH >= 1.0f) fH -= 1.0f;
    } else {
      // color is pure Black.
      fS = 0.0f;
      fH = 0.0f;  // undefined hue
    }
    return visioncpp::pixel::F32C3(fH, fS, fV);
  }
};
}
