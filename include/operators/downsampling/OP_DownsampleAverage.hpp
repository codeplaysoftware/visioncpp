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

/// \file OP_DownsampleAverage.hpp
/// \brief This file contains the downsampling filter using average technique

namespace visioncpp {
/// \struct OP_DownsampleAverage
/// \brief Downsampling filter using average technique
/// Other filters could be added for different numbers of channels.
struct OP_DownsampleAverage {
  /// \param rdn
  /// \return PIXEL
  template <typename NeighbourT>
  typename NeighbourT::PixelType operator()(NeighbourT &rdn) {
    auto out = rdn.at(2 * rdn.I_c, 2 * rdn.I_r) +
               rdn.at(2 * rdn.I_c + 1, 2 * rdn.I_r + 1) / 2.0f;
    return out;
  }
};
}
