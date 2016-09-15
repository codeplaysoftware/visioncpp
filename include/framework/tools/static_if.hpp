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

/// \file static_if.hpp
/// \brief This file provides a set of static_if functions used at to calculate
/// the result of if statement at compile time

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_TOOLS_STATIC_IF_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_TOOLS_STATIC_IF_HPP_

namespace visioncpp {
namespace internal {
namespace tools {
/// \struct IfNode
/// \brief This struct is used to determine whether or not an expression node is
/// a leafNode
/// template parameters
/// \tparam T is the type of the expression tree
template <typename T>
struct IfNode {
  static constexpr bool Is_LeafNode = false;
};
/// \brief specialisation of the IfNode when the node is leafNode
template <typename Nested, size_t LVL>
struct IfNode<LeafNode<Nested, LVL>> {
  static constexpr bool Is_LeafNode = true;
};

/// \struct StaticIf
/// \brief It is used to select either of the input type based the Conds
/// template parameters
/// \tparam Conds : determines the condition
/// \tparam T1 : first input type
/// \tparam T2 : second input type
template <bool Conds, typename T1, typename T2>
struct StaticIf {
  using Type = T1;
  using LostType = T2;
};
/// \brief specialisation of the StaticIf when the condition is false
template <typename T1, typename T2>
struct StaticIf<false, T1, T2> {
  using Type = T2;
  using LostType = T1;
};
/// \struct StaticIf
/// \brief It is used to select either of the template constant variable based
/// the Conds
/// template parameters
/// \tparam Conds : determines the condition
/// \tparam X : first constant variable
/// \tparam Y : second constant type
template <bool Conds, size_t X, size_t Y>
struct IfConst {
  static constexpr size_t Value = X;
};
/// specialisation of the IfConst when the condition is false
template <size_t X, size_t Y>
struct IfConst<false, X, Y> {
  static constexpr size_t Value = Y;
};

}  // tools
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_TOOLS_STATIC_IF_HPP_
