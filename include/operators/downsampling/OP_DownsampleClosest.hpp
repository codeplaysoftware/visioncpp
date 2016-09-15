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

/// \file OP_DownsampleClosest.hpp
/// \brief This files contains downsampling filter using closest technique
/// If the number of channel is different in your case feel free to write your
/// own.

namespace visioncpp {
/// \struct OP_DownsampleClosest
/// \brief Downsampling filter using closest technique
struct OP_DownsampleClosest {
  /// \param rdn
  /// \return PIXEL
  template <typename NeighbourT>
  typename NeighbourT::PixelType operator()(NeighbourT &rdn) {
    return rdn.at(2 * rdn.I_c, 2 * rdn.I_r);
  }
};
}
