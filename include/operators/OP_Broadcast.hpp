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

/// \file OP_Broadcast.hpp
/// \brief This file contains a struct to set pixel to the value passed in.

namespace visioncpp {
/// \struct OP_Broadcast
/// \brief This functor sets the pixel to the value passed in.
struct OP_Broadcast {
  /// \param val
  /// \return SCALAR
  template <typename SCALAR>
  SCALAR operator()(SCALAR val) {
    return val;
  }
};
}
