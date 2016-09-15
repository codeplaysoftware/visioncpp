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

/// \file expr_convertor.hpp
/// \brief this is used to replace the leaf node with a PlaceHolder node.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_EXPR_CONVERTOR_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_EXPR_CONVERTOR_HPP_

namespace visioncpp {
namespace internal {
/// \struct PlaceHolder
/// \brief PlaceHolder is used to replace the Vision Memory in the LeafNode
/// containing sycl buffer. PlaceHolder contains the order of the leaf node in
/// the expression tree.
/// template parameters:
/// \tparam Memory_Type: represents the type of the memory
/// \tparam N: represents the order of the visionMemory in the LeafNode
/// \tparam C: contains the column size of the visionMemory
/// \tparam R: contains the row size of the visionMemory
template <size_t Memory_Type, size_t N, size_t C, size_t R, size_t Sc>
struct PlaceHolder {
  static constexpr size_t I = N;
  static constexpr size_t Cols = C;
  static constexpr size_t Rows = R;
  static constexpr size_t LeafType = Memory_Type;
  using Type = PlaceHolder<LeafType, N, C, R, Sc>;
  using OutType = Type;
  static constexpr size_t Scope = Sc;
};

/// \brief LocalOutput:Local output is used for neighbour operation in order to
/// create a local memory for the output of non-terminal nodes in the expression
/// tree when the NeighbourOP is used.
template <bool PointOp, size_t IsRoot, size_t LC, size_t LR, typename Expr>
struct LocalOutput;

/// \brief is used to count the total number of leafNode in the expression tree.
template <size_t Category, typename Expr>
struct LeafCount;
/// \brief is used to count the total number of local memory for the
/// subxpression.
template <size_t Category, typename Expr>
struct LocalMemCount;

// template <size_t Memory_Type, size_t N, size_t R = 0, size_t C = 0>
// struct PlaceHolder;
/// \brief it is used to create the PlaceHolder expression. The PlaceHolder
/// expression is a copy of expression type where the visionMemory of the
/// leaf node has been replaced with PlaceHolder.
template <size_t Category, typename Expr, size_t N>
struct MakePlaceHolderExprHelper;

}  // internal
}  // visioncpp
// Static Operation Over Type
#include "leaf_count.hpp"
#include "local_mem_count.hpp"
#include "local_output.hpp"
#include "make_place_holder_expr.hpp"
#include "place_holder_leaf_node.hpp"
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_EXPR_CONVERTOR_HPP_
