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

/// \file OP_GaussianBlur3x3.hpp
/// \brief this file apply the Gaussian blur 3x3

namespace visioncpp {
/// \struct OP_GaussianBlur3x3
/// \brief applying the Gaussian blur 3x3
struct OP_GaussianBlur3x3 {
  /// \param nbr
  /// \return PIXEL
  template <typename NeighbourT>
  typename NeighbourT::PixelType operator()(NeighbourT &nbr) {
    auto out = nbr.at(nbr.I_c - 1, nbr.I_r - 1) * 1.0f / 16.0f;
    out += nbr.at(nbr.I_c - 1, nbr.I_r + 1) * 1.0f / 16.0f;
    out += nbr.at(nbr.I_c + 1, nbr.I_r - 1) * 1.0f / 16.0f;
    out += nbr.at(nbr.I_c + 1, nbr.I_r + 1) * 1.0f / 16.0f;
    out += nbr.at(nbr.I_c, nbr.I_r - 1) * 2.0f / 16.0f;
    out += nbr.at(nbr.I_c - 1, nbr.I_r) * 2.0f / 16.0f;
    out += nbr.at(nbr.I_c, nbr.I_r + 1) * 2.0f / 16.0f;
    out += nbr.at(nbr.I_c + 1, nbr.I_r) * 2.0f / 16.0f;
    out += nbr.at(nbr.I_c, nbr.I_r) * 4.0f / 16.0f;
    return out;
  }
};
}
