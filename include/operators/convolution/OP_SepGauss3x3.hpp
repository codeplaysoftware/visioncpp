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

/// \file OP_SepGauss3x3.hpp
/// \brief Separable Gaussian filter for 3x3 filter size

namespace visioncpp {
/// \struct OP_SepGaussRow3
/// \brief Separable filter for rows.
/// If the number of channel is different in your case feel free to write your
/// own.
struct OP_SepGaussRow3 {
  /// \param nbr - Where PIXEL is pixel type
  /// \return PIXEL
  template <typename NeighbourT>
  typename NeighbourT::PixelType operator()(NeighbourT &nbr) {
    auto out = nbr.at(nbr.I_c, nbr.I_r - 1) * 1.0f / 4.0f;
    out += nbr.at(nbr.I_c, nbr.I_r) * 2.0f / 4.0f;
    out += nbr.at(nbr.I_c, nbr.I_r + 1) * 1.0f / 4.0f;
    return out;
  }
};

/// \struct OP_SepGaussCol3
/// \brief Separable filter for cols.
struct OP_SepGaussCol3 {
  /// \param nbr - Where PIXEL is pixel type
  /// \return PIXEL
  template <typename NeighbourT>
  typename NeighbourT::PixelType operator()(NeighbourT &nbr) {
    auto out = nbr.at(nbr.I_c - 1, nbr.I_r) * 1.0f / 3.0f;
    out += nbr.at(nbr.I_c, nbr.I_r) * 1.0f / 3.0f;
    out += nbr.at(nbr.I_c + 1, nbr.I_r) * 1.0f / 3.0f;
    return out;
  }
};
}
