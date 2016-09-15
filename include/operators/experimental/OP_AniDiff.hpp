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

/// \file OP_AniDiff.hpp
/// \brief It applies a simplified version of the anisotropic diffusion

namespace visioncpp {
/// \brief This functor applies anisotropic diffusion for one channel
struct OP_AniDiff_Grey {
  /// \brief This functor applies a simplified version of the anisotropic
  /// diffusion for one channel
  /// \param nbr - An image with one channel
  /// \return float - Return the anisotropic diffusion in one channel
  template <typename T>
  float operator()(T nbr) {
    float out = 0;
    float sum_w = 0;

    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        float w =
            exp((-30.0f) * cl::sycl::fabs(nbr.at(nbr.I_c, nbr.I_r) -
                                          nbr.at(nbr.I_c + i, nbr.I_r + j)));
        sum_w += w;
        out += w * nbr.at(nbr.I_c + i, nbr.I_r + j);
      }
    }
    return out / sum_w;
  }
};

/// \brief This functor applies anisotropic diffusion for 3 channels
struct OP_AniDiff {
  /// \brief This functor applies a simplified version of the anisotropic
  /// diffusion for 3 channels
  /// \param nbr - An image with three channels
  /// \return float - Returns the anisotropic diffusion in 3 channels
  template <typename T>
  typename T::PixelType operator()(T nbr) {
    using Type = typename T::PixelType::data_type;

    cl::sycl::float4 out(0, 0, 0, 0);
    cl::sycl::float4 sum_w(0, 0, 0, 0);

    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        cl::sycl::float4 p1(nbr.at(nbr.I_c, nbr.I_r)[0],
                            nbr.at(nbr.I_c, nbr.I_r)[1],
                            nbr.at(nbr.I_c, nbr.I_r)[2], 0);
        cl::sycl::float4 p2(nbr.at(nbr.I_c + i, nbr.I_r + j)[0],
                            nbr.at(nbr.I_c + i, nbr.I_r + j)[1],
                            nbr.at(nbr.I_c + i, nbr.I_r + j)[2], 0);
        cl::sycl::float4 w = exp((-30.0f) * cl::sycl::fabs(p1 - p2));
        sum_w += w;
        out += w * p2;
      }
    }
    out = out / sum_w;
    return typename T::PixelType(static_cast<Type>(out.x()),
                                 static_cast<Type>(out.y()),
                                 static_cast<Type>(out.z()));
  }
};
}
