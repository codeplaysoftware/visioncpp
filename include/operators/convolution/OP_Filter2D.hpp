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

/// \file OP_Filter2D.hpp
/// \brief This file apply the general convolution

namespace visioncpp {
/// \struct OP_Filter2D
/// \brief Applying the general convolution for 3 channel Image.
/// If the number of channel is different in your case feel free to write your
/// own.
struct OP_Filter2D {
  /// \param nbr
  /// \param fltr
  /// \return NeighbourT::PixelType
  template <typename NeighbourT, typename FilterT>
  typename NeighbourT::PixelType operator()(NeighbourT& nbr, FilterT& fltr) {
    int i, i2, j, j2;
    int hs_c = (fltr.cols / 2);
    int hs_r = (fltr.rows / 2);
    typename NeighbourT::PixelType out{};
    for (i2 = -hs_c, i = 0; i2 <= hs_c; i2++, i++)
      for (j2 = -hs_r, j = 0; j2 <= hs_r; j2++, j++)
        out += nbr.at(nbr.I_c + i2, nbr.I_r + j2) * fltr.at(i, j);
    return out;
  }
};

/// \struct OP_Filter2D_One
/// \brief Applying the general convolution for 1 channel Image.
/// If the number of channel is different in your case feel free to write your
/// own.
struct OP_Filter2D_One {
  /// \param nbr
  /// \param fltr
  /// \return float
  template <typename NeighbourT, typename FilterT>
  float operator()(NeighbourT& nbr, FilterT& fltr) {
    int i, i2, j, j2;
    int hs_c = (fltr.cols / 2);
    int hs_r = (fltr.rows / 2);
    float out = 0;
    for (i2 = -hs_c, i = 0; i2 <= hs_c; i2++, i++)
      for (j2 = -hs_r, j = 0; j2 <= hs_r; j2++, j++)
        out += (nbr.at(nbr.I_c + i2, nbr.I_r + j2) * fltr.at(i, j));
    return out;
  }
};
}
