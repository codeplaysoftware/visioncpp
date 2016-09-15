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

/// \file framework.hpp
/// \brief Collection of VisionCpp framework headers.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_FRAMEWORK_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_FRAMEWORK_HPP_

// including all the claas name required by the framework
#include "forward_declarations.hpp"

// include tools headers
#include "tools/tools.hpp"

// include memory headers
#include "memory/memory.hpp"

// expression with placeholder leafnode includes
#include "expr_convertor/expr_convertor.hpp"

// Expression Tree headers
#include "expr_tree/expr_tree.hpp"

// Evaluation tree headers
#include "evaluator/evaluator.hpp"

// Executor Policies
#include "executor/executor.hpp"

// devices used to execute the code
#include "device/device.hpp"
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_FRAMEWORK_HPP_
