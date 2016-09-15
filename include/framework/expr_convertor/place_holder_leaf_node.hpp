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

/// \file place_holder_leaf_node.hpp
/// \brief This file contains different specialisations leafNodes for different
/// types of memory.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_PLACE_HOLDER_LEAF_NODE_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_PLACE_HOLDER_LEAF_NODE_HPP_

namespace visioncpp {
namespace internal {
/// The specialisation of LeafNode for the PlaceHolder. It is used to store the
/// information of visionMemory.
/// template parameters:
/// \param Memory_Type : determines the type of the memory {Buffer2D, Buffer1D,
/// Host, Image}
/// \param N: is the position of the leafNode in the expression tree
/// \param C is the column size of the visionMemory
/// \param R is the row size of the visionMemory
/// \param LVL represents the level of the node in the expression tree.
template <size_t Memory_Type, size_t N, size_t C, size_t R, size_t LVL,
          size_t Sc>
struct LeafNode<PlaceHolder<Memory_Type, N, C, R, Sc>, LVL> {
  static constexpr bool PointOp = true;
  static constexpr size_t LeafType = Memory_Type;
  using RHSExpr = PlaceHolder<LeafType, N, C, R, Sc>;
  using Type = typename RHSExpr::Type;
  using OutType = typename RHSExpr::OutType;
  static constexpr size_t RThread = Type::Rows;
  static constexpr size_t CThread = Type::Cols;
  static constexpr size_t ND_Category = internal::expr_category::Unary;
  static constexpr size_t Level = N;
  static constexpr bool SubExpressionEvaluationNeeded = false;
  static constexpr size_t Operation_type = internal::ops_category::PointOP;
  using Sub_expression_Type =
      LeafNode<PlaceHolder<Memory_Type, N, C, R, Sc>, LVL>;
  template <bool ForcedToExec, size_t LC, size_t LR, size_t LCT, size_t LRT,
            typename DeviceT>
  Sub_expression_Type inline sub_expression_evaluation(const DeviceT &);
};
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_PLACE_HOLDER_LEAF_NODE_HPP_
