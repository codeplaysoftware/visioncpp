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

/// \file OP_SepFilter.hpp
/// \brief This file applies Separable filter convolution

namespace visioncpp {
/// \struct OP_SepFilterRow
/// \brief Separable filter for rows.
/// If the number of channel is different in your case feel free to write your
/// own.
struct OP_SepFilterRow {
  /// \param nbr - where NeighbourT is pixel type
  /// \param fltr - filter array
  /// \return NeighbourT::PixelType
  template <typename NeighbourT, typename FilterT>
  typename NeighbourT::PixelType operator()(NeighbourT& nbr, FilterT& fltr) {
    int i, i2;
    int hs_r = (fltr.rows / 2);
    auto out = nbr.at(nbr.I_c, nbr.I_r - 1) * fltr.at(0, 0);
    for (i2 = -hs_r + 1, i = 1; i2 <= hs_r; i2++, i++) {
      out += nbr.at(nbr.I_c, nbr.I_r + i2) * fltr.at(0, i);
    }
    return out;
  }
};

/// \struct OP_SepFilterCol
/// \brief Separable filter for cols.
struct OP_SepFilterCol {
  /// \param nbr - where NeighbourT is pixel type
  /// \param fltr - filter array
  /// \return NeighbourT::PixelType
  template <typename NeighbourT, typename FilterT>
  typename NeighbourT::PixelType operator()(NeighbourT& nbr, FilterT& fltr) {
    int i, i2;
    int hs_c = (fltr.cols / 2);
    auto out = nbr.at(nbr.I_c - 1, nbr.I_r) * fltr.at(0, 0);
    for (i2 = -hs_c + 1, i = 1; i2 <= hs_c; i2++, i++) {
      out += nbr.at(nbr.I_c + i2, nbr.I_r) * fltr.at(i, 0);
    }
    return out;
  }
};
}
