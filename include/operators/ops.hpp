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

/// \file ops.hpp
/// \brief This header gathers all operations available in VisionCpp.

#ifndef VISIONCPP_INCLUDE_OPERATORS_OPS_HPP_
#define VISIONCPP_INCLUDE_OPERATORS_OPS_HPP_

// supported operators - covered by a testcase
#include "convert/ops_convert.hpp"
#include "convolution/ops_conv.hpp"
#include "downsampling/ops_downsampling.hpp"
// interop with openCV
#include "opencvinterop.hpp"

#include "OP_Broadcast.hpp"
#include "OP_ScaleChannel.hpp"

#include "experimental/experimental.hpp"
#endif  // VISIONCPP_INCLUDE_OPERATORS_OPS_HPP_