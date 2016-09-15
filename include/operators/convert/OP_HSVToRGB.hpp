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

/// \file OP_HSVToRGB.hpp
/// \brief it converts HSV pixel to RGB pixel

namespace visioncpp {
/// \brief functor converts HSV(H:[0.0f, 360.f], S:[0.0f, 1.0f] V: [0.0f, 1.0f])
/// to  color RGB(R:[0.0f, 1.0f], G:[0.0f, 1.0f], B:[0.0f, 1.0f])
struct OP_HSVToRGB {
  /// \param inHSV
  /// \return RGB
  visioncpp::pixel::F32C3 operator()(visioncpp::pixel::F32C3 inHSV) {
    float fR = 0.0f, fG = 0.0f, fB = 0.0f;
    float fH = inHSV[0];  // H component
    float fS = inHSV[1];  // S component
    float fV = inHSV[2];  // V component

    // Convert from HSV to RGB, using float ranges 0.0 to 1.0
    int iI = 0;
    float fI = 0.0f, fF = 0.0f, p = 0.0f, q = 0.0f, t = 0.0f;

    if (fS == 0.0f) {
      // achromatic (grey)
      fR = fG = fB = fV;
    } else {
      // If Hue == 1.0, then wrap it around the circle to 0.0
      if (fH >= 1.0f) fH = 0.0f;

      fH *= 6.0f;                 // sector 0 to 5
      fI = cl::sycl::floor(fH);   // integer part of h (0,1,2,3,4,5 or 6)
      iI = static_cast<int>(fH);  //		"		" "      "
      fF = fH - fI;               // factorial part of h (0 to 1)

      p = fV * (1.0f - fS);
      q = fV * (1.0f - fS * fF);
      t = fV * (1.0f - fS * (1.0f - fF));

      switch (iI) {
        case 0:
          fR = fV;
          fG = t;
          fB = p;
          break;
        case 1:
          fR = q;
          fG = fV;
          fB = p;
          break;
        case 2:
          fR = p;
          fG = fV;
          fB = t;
          break;
        case 3:
          fR = p;
          fG = q;
          fB = fV;
          break;
        case 4:
          fR = t;
          fG = p;
          fB = fV;
          break;
        default:  // case 5 (or 6):
          fR = fV;
          fG = p;
          fB = q;
          break;
      }
    }

    return visioncpp::pixel::F32C3(fR, fG, fB);
  }
};
}
